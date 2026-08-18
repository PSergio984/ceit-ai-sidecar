"""Build the LLM-as-judge question set from the CURRENT bundled catalog.

Reproducible (seeded RNG). Catalog-only by design: the bundled policy corpus
is a synthetic placeholder (Faker-Latin regulation text), so policy Q&A
cannot be answered — including it would only manufacture NON_RELEVANT
verdicts. The golden retrieval set (scripts/build_golden.py) has the same
catalog-only scope.

Usage:
    uv run python scripts/build_judge_questions.py [--out data/judge_questions.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "corpus" / "catalog.json"
OUT_DEFAULT = ROOT / "data" / "judge_questions.json"

# Title Qs / code Qs / author Qs / dept-year Qs / what-is-it Qs.
SHAPES = {
    "title_code": 10,
    "code_paper": 10,
    "author_papers": 8,
    "dept_year": 6,
    "title_about": 6,
}


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw["documents"]


def author_name(doc: dict) -> str:
    authors = doc["metadata"].get("authors") or []
    if not authors:
        return ""
    name = authors[0]
    for prefix in ("Prof.", "Mr.", "Mrs.", "Ms.", "Engr.", "Dr."):
        if name.startswith(prefix + " "):
            return name[len(prefix) + 1 :]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    docs = load_catalog()
    rng = random.Random(args.seed)
    pool = docs.copy()
    rng.shuffle(pool)

    token_docs: dict[str, int] = {}
    for doc in docs:
        for token in set(re.findall(r"[a-z0-9]+", doc["title"].lower())):
            token_docs[token] = token_docs.get(token, 0) + 1
    top_tokens = {t for t, _ in sorted(token_docs.items(), key=lambda kv: kv[1], reverse=True)[:25]}

    def discriminative(doc: dict) -> bool:
        tokens = re.findall(r"[a-z0-9]+", doc["title"].lower())
        return (
            len(tokens) >= 2
            and not any(t in top_tokens for t in tokens)
            and sum(1 for t in tokens if token_docs.get(t, 0) <= 5) >= 2
        )

    questions: list[dict] = []
    qid = 0

    def add(question: str, corpus: str = "catalog") -> None:
        nonlocal qid
        qid += 1
        questions.append({"id": f"q{qid:02d}", "question": question, "corpus": corpus})

    title_pool = [d for d in pool if discriminative(d)]

    # 1. "What is the catalog code for <title>?" — title -> code lookup.
    for doc in title_pool[: SHAPES["title_code"]]:
        add(f"What is the catalog code for {doc['title'].strip()}?")

    # 2. "<code> is what paper?" — code -> title lookup.
    for doc in title_pool[SHAPES["title_code"] : SHAPES["title_code"] + SHAPES["code_paper"]]:
        add(f"{doc['metadata']['catalog_code']} is what paper?")

    # 3. "What papers did <author> write?" — name lookup.
    authors = [d for d in pool if author_name(d)][: SHAPES["author_papers"]]
    for doc in authors:
        add(f"What papers did {author_name(doc)} write?")

    # 4. dept + year combos with a modest answer set.
    depts = sorted({d["metadata"]["department"] for d in docs})
    years = sorted({d["metadata"]["publication_year"] for d in docs})
    used: set[tuple] = set()
    for _ in range(400):
        dept, year = rng.choice(depts), rng.choice(years)
        size = sum(
            1
            for d in docs
            if d["metadata"]["department"] == dept and d["metadata"]["publication_year"] == year
        )
        if 5 <= size <= 60 and (dept, year) not in used:
            used.add((dept, year))
            add(f"What papers were published by the {dept} department in {year}?")
            if len(used) >= SHAPES["dept_year"]:
                break

    # 5. "What is <title> about?" — title re-asking (doc text carries the answer).
    for doc in title_pool[SHAPES["title_code"] + SHAPES["code_paper"] :][: SHAPES["title_about"]]:
        add(f"What is {doc['title'].strip()} about?")

    dataset = {
        "version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "corpus": {"file": "corpus/catalog.json", "documents": len(docs)},
        "questions": questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.out} — {len(questions)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
