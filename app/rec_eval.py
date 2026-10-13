"""Offline leave-last-out evaluation of the recommender on borrow sequences.

Input is the Laravel `ai:export-borrow-sequences` export (schema_version 1):
per anonymised user an ordered list of academic_paper ids, oldest first.
For every user with >=2 distinct papers the last distinct paper is held out
as the target and the rest form the prefix; each approach ranks the catalog
with the prefix excluded. Metrics: Hit@K (target inside the top K) and MRR
over the top-LIST_SIZE list.

Pure logic — the catalog artifacts and the title ranker are injected so the
embedding model is never a module-import or test dependency.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fusion import recommend_rank

SCHEMA_VERSION = 1
# Same caps as production Personal recommendations and the /recommend contract.
SEED_COUNT = 10
LIST_SIZE = 50
# Below this many evaluated users the numbers are noise, not evidence.
MIN_ELIGIBLE = 20

APPROACHES = ("recommend", "popularity", "title-as-query")


@dataclass
class EvalCase:
    seeds: list[str]  # last up-to-SEED_COUNT prefix papers, paper-N form
    exclude_ids: list[str]  # every prefix paper, paper-N form
    prefix: list[int]  # distinct prefix paper ids, first-occurrence order
    target: int  # held-out paper id


@dataclass
class EvalStats:
    total_users: int
    eligible_users: int
    ineligible_users: int
    unknown_paper_skips: int


def load_sequences(path: Path | str) -> dict[str, list[int]]:
    """Parse a borrow-sequence export; ValueError on a bad file or schema."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read borrow-sequence export {path}: {exc}") from exc
    version = payload.get("schema_version") if isinstance(payload, dict) else None
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    sequences = payload.get("sequences")
    try:
        return {key: [int(i) for i in seq] for key, seq in sequences.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed sequences: {exc}") from exc


def _dedupe(seq: list[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for i in seq:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_cases(
    sequences: dict[str, list[int]], catalog_ids: set[str]
) -> tuple[list[EvalCase], EvalStats]:
    """Leave-last-out split: last distinct paper held out, the rest prefix.

    A user needs >=2 distinct papers to produce a case. Papers whose paper-N
    id is not in the catalog are dropped and counted; a user whose held-out
    target is unknown (or whose whole prefix is unknown) is ineligible —
    the case cannot be scored.
    """
    cases: list[EvalCase] = []
    ineligible = 0
    unknown = 0
    for seq in sequences.values():
        distinct = _dedupe(seq)
        if len(distinct) < 2:
            ineligible += 1
            continue
        target = distinct[-1]
        if f"paper-{target}" not in catalog_ids:
            ineligible += 1
            unknown += 1
            continue
        prefix = []
        for pid in distinct[:-1]:
            if f"paper-{pid}" in catalog_ids:
                prefix.append(pid)
            else:
                unknown += 1
        if not prefix:
            ineligible += 1
            continue
        cases.append(
            EvalCase(
                seeds=[f"paper-{p}" for p in prefix[-SEED_COUNT:]],
                exclude_ids=[f"paper-{p}" for p in prefix],
                prefix=prefix,
                target=target,
            )
        )
    stats = EvalStats(
        total_users=len(sequences),
        eligible_users=len(cases),
        ineligible_users=ineligible,
        unknown_paper_skips=unknown,
    )
    return cases, stats


def _borrow_counts(sequences: dict[str, list[int]], catalog_ids: set[str]) -> Counter:
    """Borrow counts over prefix papers only — an eligible user's held-out
    target never counts, so a popular target cannot leak into the ranking."""
    counts: Counter = Counter()
    for seq in sequences.values():
        distinct = _dedupe(seq)
        held_out = distinct[-1] if len(distinct) >= 2 else None
        for pid in seq:
            if pid != held_out and f"paper-{pid}" in catalog_ids:
                counts[pid] += 1
    return counts


def popularity_ranking(counts: Counter) -> list[str]:
    """paper-N ids ordered by borrow count desc, paper id asc on ties."""
    return [f"paper-{p}" for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def is_hit(ranked_ids: list[str], target_id: str, k: int) -> bool:
    return target_id in ranked_ids[:k]


def reciprocal_rank(ranked_ids: list[str], target_id: str) -> float:
    """1-based reciprocal rank of the target; 0.0 when absent."""
    try:
        return 1.0 / (ranked_ids.index(target_id) + 1)
    except ValueError:
        return 0.0


def evaluate(
    docs_by_id: dict,
    vectors: np.ndarray,
    sequences: dict[str, list[int]],
    title_ranker: Callable[[str], list[str]],
    hit_at: int = 10,
    list_size: int = LIST_SIZE,
) -> dict:
    """Run all three approaches over the leave-last-out cases.

    `title_ranker` maps a query title to ranked catalog doc ids — the seam
    for the title-as-query baseline (HybridSearch.rrf_search in the CLI, a
    deterministic stub in tests).
    """
    if hit_at < 1 or hit_at > list_size:
        raise ValueError(f"hit_at must be between 1 and list_size ({list_size})")
    catalog_ids = {did for did, d in docs_by_id.items() if d.get("corpus") == "catalog"}
    cases, stats = build_cases(sequences, catalog_ids)
    popularity = popularity_ranking(_borrow_counts(sequences, catalog_ids))

    hits: Counter = Counter()
    rr: Counter = Counter()
    for case in cases:
        excluded = set(case.exclude_ids)
        target_id = f"paper-{case.target}"
        _, rec_results = recommend_rank(
            docs_by_id, vectors, case.seeds, case.exclude_ids, list_size
        )
        title = docs_by_id[f"paper-{case.prefix[-1]}"].get("title") or ""
        ranked = {
            "recommend": [r["id"] for r in rec_results],
            "popularity": [pid for pid in popularity if pid not in excluded][:list_size],
            "title-as-query": [
                pid for pid in title_ranker(title) if pid not in excluded and pid in catalog_ids
            ][:list_size],
        }
        for approach, ids in ranked.items():
            if is_hit(ids, target_id, hit_at):
                hits[approach] += 1
            rr[approach] += reciprocal_rank(ids[:list_size], target_id)

    n = len(cases)
    return {
        "stats": stats,
        "hit_at": hit_at,
        "results": {
            name: {
                "hit_at_k": hits[name] / n if n else 0.0,
                "mrr": rr[name] / n if n else 0.0,
            }
            for name in APPROACHES
        },
    }


def format_report(outcome: dict, source: str) -> str:
    """Plain-text table; thin data is labelled informational, never hidden."""
    stats: EvalStats = outcome["stats"]
    lines = [
        f"Borrow-sequence recommender evaluation ({source})",
        f"Users in file                         : {stats.total_users}",
        f"Eligible users (>=2 distinct borrows) : {stats.eligible_users}",
        f"Unknown-paper skips                   : {stats.unknown_paper_skips}",
        "",
        f"{'Approach':<16} {'Hit@' + str(outcome['hit_at']):>8} {'MRR':>8}",
    ]
    for name in APPROACHES:
        r = outcome["results"][name]
        lines.append(f"{name:<16} {r['hit_at_k']:>8.3f} {r['mrr']:>8.3f}")
    if stats.eligible_users < MIN_ELIGIBLE:
        lines.append("")
        lines.append(
            f"INFORMATIONAL ONLY: fewer than {MIN_ELIGIBLE} eligible users ({stats.eligible_users})"
        )
    return "\n".join(lines)
