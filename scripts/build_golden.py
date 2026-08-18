"""Build the golden evaluation set from the CURRENT bundled corpus.

Reproducible: seeded RNG, deterministic queries derived from the corpus
itself. The previous set was built against a 48-doc synthetic snapshot and
scores ~0 against the 1,338-doc production export, so the set must track
the shipped data — regenerate it whenever the corpus changes.

Taxonomy (27 cases, mirrored from the original set):
  - catalog_code (4): the catalog code verbatim
  - exact_title (8): the paper title verbatim
  - paraphrase  (6): natural student phrasings with a 5-30 doc answer set
  - people      (4): "papers by <author>"
  - negative    (5): out-of-domain queries that must return nothing

Usage:
    uv run python scripts/build_golden.py [--out data/golden_dataset.json]
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
OUT_DEFAULT = ROOT / "data" / "golden_dataset.json"

NEGATIVES = [
    "recipes for a birthday cake",
    "how to fix a car engine",
    "best travel destinations in japan",
    "mathematical proofs about prime numbers",
    "soccer training drills for beginners",
]

# (template, filter spec). The relevant set is computed from the spec ONLY —
# a query that mentions just the year must not count dept+type+year docs.
# Combos are chosen so the true answer set lands in [TARGET_MIN, TARGET_MAX]
# on this corpus (dept-only ~320 docs, type-only ~160, type x dept ~53).
PHRASE_TEMPLATES = [
    (
        "papers from the {department} department published in {year}",
        ["department", "publication_year"],
    ),
    ("{paper_type} papers from {year}", ["paper_type", "publication_year"]),
    ("engineering papers published in {year}", ["publication_year"]),
    ("recent {department} research papers from {year}", ["department", "publication_year"]),
    (
        "research papers from the {department} department in {year}",
        ["department", "publication_year"],
    ),
    ("{paper_type} papers from the {department} department", ["paper_type", "department"]),
]

# "Feasib" is the truncated export value — spell it out in queries.
TYPE_LABELS = {"Feasib": "Feasibility"}

# Acceptable size of the true relevant set for a paraphrase query.
TARGET_MIN, TARGET_MAX = 5, 60


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw if isinstance(raw, list) else raw["documents"]


def doc_matches(doc: dict, **filters) -> bool:
    meta = doc.get("metadata", {})
    for key, value in filters.items():
        if key == "year_from":
            if int(meta.get("publication_year") or 0) < value:
                return False
        elif meta.get(key) != value:
            return False
    return True


def ids_for(docs: list[dict], **filters) -> list[str]:
    return [d["id"] for d in docs if doc_matches(d, **filters)]


def pick_filters(docs: list[dict], rng: random.Random, spec: list[str]) -> dict | None:
    """Sample values for a template's filter spec whose doc count is in band.

    `year_from` is deterministic: the latest year minus one (a "recent"
    qualifier). Values use METADATA keys so they plug straight into
    ``ids_for``; the template's own placeholders are resolved separately.
    """
    values = {
        "department": sorted({d["metadata"].get("department") for d in docs}),
        "paper_type": sorted({d["metadata"].get("paper_type") for d in docs}),
        "publication_year": sorted({d["metadata"].get("publication_year") for d in docs}),
    }
    recent_year = max(values["publication_year"]) - 1
    for _ in range(300):
        filters = {}
        for key in spec:
            filters[key] = recent_year if key == "year_from" else rng.choice(values[key])
        size = len(ids_for(docs, **filters))
        if TARGET_MIN <= size <= TARGET_MAX:
            return filters
    return None


def format_query(template: str, filters: dict) -> str:
    """Fill template placeholders from the sampled metadata values."""
    mapping = {
        "department": filters.get("department"),
        "paper_type": TYPE_LABELS.get(filters.get("paper_type"), filters.get("paper_type")),
        "year": filters.get("publication_year"),
    }
    return template.format(**mapping).strip()


def author_name(doc: dict) -> str:
    authors = doc["metadata"].get("authors") or []
    if not authors:
        return ""
    name = authors[0]
    # Strip the honorific ("Prof. Emmanuelle Bechtelar II" -> "Emmanuelle Bechtelar II")
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
    by_id = {d["id"]: d for d in docs}
    rng = random.Random(args.seed)
    cases: list[dict] = []

    def add(query: str, relevant: list[str], negative: bool = False) -> None:
        missing = [r for r in relevant if r not in by_id]
        if missing:
            raise SystemExit(f"relevant ids missing from corpus: {missing}")
        case = {"query": query, "corpus": "catalog", "relevant_docs": relevant}
        if negative:
            case["negative"] = True
        cases.append(case)

    pool = docs.copy()
    rng.shuffle(pool)

    # Title-token corpus frequency: how many docs share each word of a title.
    # Discriminative titles (rare tokens) make exact_title actually test
    # top-1; shared-Latin titles collide with siblings and fail on ranking
    # noise instead of retrieval quality.
    token_docs: dict[str, int] = {}
    for doc in docs:
        for token in set(re.findall(r"[a-z0-9]+", doc["title"].lower())):
            token_docs[token] = token_docs.get(token, 0) + 1

    # The top-N tokens are Faker-Latin function words ("aut", "sit", "quis")
    # shared by hundreds of titles — a discriminative title contains none of
    # them and at least two genuinely rare words (e.g. "portal", "energy").
    top_tokens = {t for t, _ in sorted(token_docs.items(), key=lambda kv: kv[1], reverse=True)[:25]}

    def discriminative(doc: dict) -> bool:
        tokens = re.findall(r"[a-z0-9]+", doc["title"].lower())
        if len(tokens) < 2:
            return False
        if any(t in top_tokens for t in tokens):
            return False
        rare = sum(1 for t in tokens if token_docs.get(t, 0) <= 5)
        return rare >= 2

    # 1. catalog_code (4)
    for doc in pool[:4]:
        code = doc["metadata"].get("catalog_code")
        if not code:
            raise SystemExit(f"doc {doc['id']} has no catalog_code")
        add(code, [doc["id"]])

    # 2. exact_title (8) — discriminative titles (rare tokens) first
    title_docs = [d for d in pool[4:] if discriminative(d)]
    for doc in title_docs[:8]:
        add(doc["title"].strip(), [doc["id"]])
    if not title_docs:
        raise SystemExit("no discriminative titles found for exact_title cases")

    # 3. people (4) — "papers by X" is relevant to EVERY paper by X, not just
    # the sampled one (names repeat across papers, e.g. "Dr. Jolie Hahn Sr.").
    def name_matches(doc: dict, name: str) -> bool:
        for author in doc["metadata"].get("authors") or []:
            if author_name({"metadata": {"authors": [author]}}) == name:
                return True
        return False

    people_docs = [d for d in pool[12:16] if author_name(d)]
    for doc in people_docs:
        name = author_name(doc)
        add(f"papers by {name}", [d["id"] for d in docs if name_matches(d, name)])

    # 4. paraphrase (6) — combos with a 5-30 doc answer set
    used_filters: list[dict] = []
    for i in range(6):
        template, spec = PHRASE_TEMPLATES[i % len(PHRASE_TEMPLATES)]
        filters = pick_filters(docs, rng, spec)
        if filters is None:
            raise SystemExit(f"could not find a {spec} combo in the size band")
        query = format_query(template, filters)
        if query in {c["query"] for c in cases} or filters in used_filters:
            continue
        used_filters.append(filters)
        add(query, ids_for(docs, **filters))
    if len(used_filters) < 6:
        raise SystemExit("paraphrase generation under-produced")

    # 5. negative (5)
    for topic in NEGATIVES:
        add(topic, [], negative=True)

    dataset = {
        "version": 3,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "corpus": {"file": "corpus/catalog.json", "documents": len(docs)},
        "catalog_snapshot": [
            {
                "id": d["id"],
                "corpus": d.get("corpus", "catalog"),
                "title": d["title"],
                "metadata": d["metadata"],
            }
            for d in docs
        ],
        "test_cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)

    from collections import Counter

    kinds = Counter(
        "negative"
        if c.get("negative")
        else (
            "catalog_code"
            if c["query"].lower().startswith("ceit-")
            else (
                "people"
                if "paper" in c["query"].lower() and " by " in c["query"].lower()
                else ("exact_title" if len(c["relevant_docs"]) == 1 else "paraphrase")
            )
        )
        for c in cases
    )
    print(f"wrote {args.out} — {len(cases)} cases: {dict(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
