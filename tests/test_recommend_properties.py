"""Corpus-level property checks for recommend_rank on the committed index.

Runs the pure ranking function against the real index/ artifacts (docs-1.json
+ vectors-1.npy) — no embedding model, no network. Thresholds are measured
on index v1 (959 catalog docs across 3 departments) with headroom:

- same-department seeds: measured share 1.0 in the top-10 for every
  department -> assert >= 0.8.
- mixed-department seeds: arbitrary same-dept mixes collapse to a single
  cluster on this corpus, so seeds are drawn as the mutually closest
  cross-department pairs (a borrower whose history genuinely straddles a
  department boundary). Measured max single-department share 0.6-0.7 with
  both seed departments present -> assert <= 0.8.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from app.fusion import recommend_rank
from app.search import _version_artifacts

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
TOP_K = 10
SEEDS_PER_DEPT = 5

SAME_DEPT_MIN_SHARE = 0.8
MIXED_DEPT_MAX_SHARE = 0.8


def _paper_num(doc_id: str) -> int:
    return int(doc_id.split("-", 1)[1])


def _cross_department_seeds(
    a_ids: list[str], b_ids: list[str], index: dict[str, int], vectors: np.ndarray, k: int
) -> list[str]:
    """k seeds per department: the k mutually closest cross-department pairs.

    Deterministic greedy matching on cosine (ties break by row position) —
    picks the papers that sit closest to the other department, which is where
    a genuinely mixed borrow history lives.
    """
    va = vectors[[index[d] for d in a_ids]]
    vb = vectors[[index[d] for d in b_ids]]
    sim = va @ vb.T
    pairs = sorted(
        ((float(sim[i, j]), i, j) for i in range(len(a_ids)) for j in range(len(b_ids))),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    seeds_a, seeds_b, used_a, used_b = [], [], set(), set()
    for _, i, j in pairs:
        if len(seeds_a) >= k:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        seeds_a.append(a_ids[i])
        seeds_b.append(b_ids[j])
    return seeds_a + seeds_b


@pytest.fixture(scope="module")
def catalog():
    _, docs_by_id, vectors = _version_artifacts(INDEX_DIR)
    index = {doc_id: i for i, doc_id in enumerate(docs_by_id)}
    by_dept = defaultdict(list)
    for doc_id, doc in docs_by_id.items():
        if doc.get("corpus") == "catalog":
            by_dept[(doc.get("metadata") or {}).get("department")].append(doc_id)
    for ids in by_dept.values():
        ids.sort(key=_paper_num)
    return docs_by_id, vectors, index, by_dept


def test_same_department_seeds_yield_same_department_results(catalog):
    docs_by_id, vectors, _, by_dept = catalog
    for dept, ids in sorted(by_dept.items()):
        seeds = ids[:SEEDS_PER_DEPT]
        seeds_used, results = recommend_rank(docs_by_id, vectors, seeds, [], TOP_K)
        assert seeds_used == len(seeds)
        share = sum(1 for r in results if (r["metadata"] or {}).get("department") == dept) / len(
            results
        )
        assert share >= SAME_DEPT_MIN_SHARE, f"{dept}: share {share}"


def test_seeds_and_exclude_ids_are_never_returned(catalog):
    docs_by_id, vectors, _, by_dept = catalog
    depts = sorted(by_dept)
    seed_sets = [
        by_dept[depts[0]][:3],
        by_dept[depts[0]][:3] + by_dept[depts[1]][:3],
        by_dept[depts[2]][:10],
    ]
    excludes = by_dept[depts[0]][3:13] + by_dept[depts[1]][10:20]
    for seeds in seed_sets:
        _, results = recommend_rank(docs_by_id, vectors, seeds, excludes, 50)
        ids = {r["id"] for r in results}
        assert ids.isdisjoint(set(seeds) | set(excludes))


def test_mixed_department_seeds_are_not_dominated(catalog):
    docs_by_id, vectors, index, by_dept = catalog
    for a, b in itertools.combinations(sorted(by_dept), 2):
        seeds = _cross_department_seeds(by_dept[a], by_dept[b], index, vectors, SEEDS_PER_DEPT)
        seeds_used, results = recommend_rank(docs_by_id, vectors, seeds, [], TOP_K)
        assert seeds_used == 2 * SEEDS_PER_DEPT
        counts = Counter((r["metadata"] or {}).get("department") for r in results)
        top_dept, top_n = counts.most_common(1)[0]
        assert top_n / len(results) <= MIXED_DEPT_MAX_SHARE, f"{a}/{b}: {top_dept} {top_n}/10"
        assert a in counts and b in counts, f"{a}/{b}: {dict(counts)}"


def test_recommend_rank_is_deterministic_and_seed_order_invariant(catalog):
    docs_by_id, vectors, _, by_dept = catalog
    seeds = [did for dept in sorted(by_dept) for did in by_dept[dept][:3]]
    _, first = recommend_rank(docs_by_id, vectors, seeds, [], TOP_K)
    _, second = recommend_rank(docs_by_id, vectors, seeds, [], TOP_K)
    _, shuffled = recommend_rank(docs_by_id, vectors, seeds[::-1], [], TOP_K)
    first_ids = [r["id"] for r in first]
    assert [r["id"] for r in second] == first_ids
    assert [r["id"] for r in shuffled] == first_ids
