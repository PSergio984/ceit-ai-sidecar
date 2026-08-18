"""LLM-as-judge answer evaluation (deliverable D3).

Samples a question set, generates grounded answers through the same
RagService path the app uses, and scores each answer RELEVANT /
PARTLY_RELEVANT / NON_RELEVANT with a judge LLM. No extra model and no
labeled dataset — the configured provider is both the answerer and the
judge (the peer-review standard for the course). Results are recorded to
``data/judge_results.json`` and summarized to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import settings
from .rag import MAX_DOC_CHARS, RagService
from .search import RRF_K, HybridSearch

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

JUDGE_LABELS = ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT")

JUDGE_SYSTEM = "You are a strict, fair answer-quality judge for a university library assistant."

JUDGE_PROMPT = (
    "Evaluate how well the assistant's answer addresses the user's question, "
    "using the retrieved documents as ground truth.\n\n"
    "Question: {question}\n\n"
    "Assistant answer:\n{answer}\n\n"
    "Retrieved documents:\n{docs}\n\n"
    "Rate the answer with exactly one label:\n"
    "- RELEVANT: the answer directly and correctly answers the question.\n"
    "- PARTLY_RELEVANT: the answer addresses part of the question or is only "
    "tangentially related.\n"
    "- NON_RELEVANT: the answer does not address the question, or refuses "
    "despite relevant documents being available.\n\n"
    "Label:"
)

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "judge_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "judge_results.json"


def classify(payload: str) -> str:
    """Map a judge-model response to a contract label.

    NON_RELEVANT is checked first because it CONTAINS the substring RELEVANT
    ("NON_RELEVANT" / "NOT RELEVANT"); then PARTLY (which also contains
    RELEVANT); a bare RELEVANT match wins last; anything else defaults to
    NON_RELEVANT.
    """
    text = (payload or "").upper().replace("_", " ")
    if "NON RELEVANT" in text or "NOT RELEVANT" in text:
        return "NON_RELEVANT"
    if "PARTLY" in text:
        return "PARTLY_RELEVANT"
    if "RELEVANT" in text:
        return "RELEVANT"
    return "NON_RELEVANT"


def build_docs_summary(docs: list[dict], top_n: int = 5) -> str:
    """Numbered, truncated document summary for the judge's ground-truth context."""
    blocks = []
    for i, doc in enumerate(docs[:top_n], start=1):
        text = (doc.get("text") or doc.get("title") or "").strip().replace("\n", " ")
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS].rstrip() + "…"
        blocks.append(f"{i}. {doc.get('title', '')} - {text}")
    return "\n".join(blocks)


class LLMJudge:
    """Scores one (question, answer, docs) triple with the judge LLM."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        self._client = client
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or 32

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> OpenAI:
        from .llm import ensure_openai_client

        self._client = ensure_openai_client(self._client, self._base_url, self._api_key)
        return self._client

    def judge(self, question: str, answer: str, docs: list[dict]) -> dict:
        """Return {"label": ..., "raw": ...} for one answer."""
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, answer=answer, docs=build_docs_summary(docs)
                    ),
                },
            ],
            max_tokens=self._max_tokens,
            temperature=0,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        return {"label": classify(raw), "raw": raw}


def aggregate_judge_results(records: list[dict]) -> dict:
    """Distribution + relevance rates over judged records (pure, testable)."""
    counts = {label: 0 for label in JUDGE_LABELS}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    total = len(records)
    relevant = counts["RELEVANT"]
    partly = counts["PARTLY_RELEVANT"]
    return {
        "counts": counts,
        "total": total,
        "relevant_rate": round(relevant / total, 4) if total else 0.0,
        "partly_or_better_rate": round((relevant + partly) / total, 4) if total else 0.0,
    }


def run_judge(
    questions: list[dict],
    engine: HybridSearch,
    rag: RagService,
    judge: LLMJudge,
    top_k: int = 5,
    out: Path | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Answer + judge every question; optionally write the recorded results."""
    records: list[dict] = []
    for question in questions:
        results = engine.rrf_search(
            question["question"],
            k=RRF_K,
            limit=top_k,
            corpus=question.get("corpus"),
            include_text=True,
        )
        answer = rag.answer(question["question"], results, mode="citations")
        judged = judge.judge(question["question"], answer, results)
        record = {
            "id": question["id"],
            "question": question["question"],
            "corpus": question.get("corpus"),
            "retrieved": [r["id"] for r in results],
            "answer": answer,
            "label": judged["label"],
        }
        records.append(record)
        if not quiet:
            print(f"  [{judged['label']:<15}] {question['question'][:70]}")

    if out is not None:
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": judge.model,
            "top_k": top_k,
            "questions_answered": len(records),
            **aggregate_judge_results(records),
            "questions": records,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if not quiet:
            print(f"Recorded results to {out}")

    return records


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["questions"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge answer evaluation (RELEVANT/PARTLY_RELEVANT/NON_RELEVANT)"
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions")
    parser.add_argument(
        "--sample", type=int, default=0, help="randomly sample N questions (--seed)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-write", action="store_true", help="do not write results file")
    parser.add_argument("--json", action="store_true", help="emit only the JSON summary")
    args = parser.parse_args()

    if args.limit < 0 or args.sample < 0:
        print("--limit and --sample must be non-negative", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k must be at least 1", file=sys.stderr)
        return 2

    questions = load_questions(args.questions)
    if args.sample:
        questions = random.Random(args.seed).sample(questions, min(args.sample, len(questions)))
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        print("no questions to evaluate", file=sys.stderr)
        return 2

    engine = HybridSearch(Path(settings.cache_dir), settings.model_name)
    rag = RagService()
    judge = LLMJudge()

    if not args.json:
        print(
            f"LLM-as-judge: {len(questions)} questions | answerer+judge model "
            f"{judge.model} | top_k={args.top_k}"
        )
    records = run_judge(
        questions,
        engine,
        rag,
        judge,
        top_k=args.top_k,
        out=None if args.no_write else args.out,
        quiet=args.json,
    )
    summary = aggregate_judge_results(records)

    if args.json:
        print(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "model": judge.model,
                    "top_k": args.top_k,
                    **summary,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"\nJudged {summary['total']} answers:")
    for label in JUDGE_LABELS:
        print(f"  {label:<16} {summary['counts'][label]}")
    print(f"Relevant rate:          {summary['relevant_rate']}")
    print(f"Partly-or-better rate:  {summary['partly_or_better_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
