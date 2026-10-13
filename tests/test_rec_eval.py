"""Unit tests for app.rec_eval: leave-last-out split, metrics, baselines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.rec_eval import (
    EvalStats,
    _borrow_counts,
    build_cases,
    evaluate,
    format_report,
    is_hit,
    load_sequences,
    popularity_ranking,
    reciprocal_rank,
)


def _write_export(tmp_path, payload) -> str:
    path = tmp_path / "borrow-sequences.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _export(sequences: dict[str, list[int]]) -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-08-20T00:00:00+00:00",
        "user_count": len(sequences),
        "sequences": sequences,
    }


def test_load_sequences_reads_export_shape(tmp_path):
    path = _write_export(tmp_path, _export({"0": [12, 5, 33], "1": [7]}))
    assert load_sequences(path) == {"0": [12, 5, 33], "1": [7]}


def test_load_sequences_rejects_wrong_schema_version(tmp_path):
    payload = _export({"0": [1, 2]})
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        load_sequences(_write_export(tmp_path, payload))


def test_load_sequences_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_sequences(tmp_path / "nope.json")


def test_load_sequences_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        load_sequences(path)


CATALOG = {f"paper-{i}" for i in range(1, 40)}


def test_build_cases_dedupes_first_occurrence_and_holds_out_last():
    cases, stats = build_cases({"0": [5, 3, 5, 9]}, CATALOG)
    assert stats == EvalStats(
        total_users=1, eligible_users=1, ineligible_users=0, unknown_paper_skips=0
    )
    assert cases[0].target == 9
    assert cases[0].prefix == [5, 3]
    assert cases[0].seeds == ["paper-5", "paper-3"]
    assert cases[0].exclude_ids == ["paper-5", "paper-3"]


def test_build_cases_ineligible_with_fewer_than_two_distinct():
    cases, stats = build_cases({"0": [5, 5, 5], "1": [1, 2]}, CATALOG)
    assert [c.target for c in cases] == [2]
    assert stats.eligible_users == 1
    assert stats.ineligible_users == 1
    assert stats.total_users == 2


def test_build_cases_caps_seeds_at_ten_but_excludes_every_prefix_paper():
    cases, _ = build_cases({"0": list(range(1, 13))}, CATALOG)
    assert cases[0].target == 12
    assert cases[0].seeds == [f"paper-{i}" for i in range(2, 12)]
    assert cases[0].exclude_ids == [f"paper-{i}" for i in range(1, 12)]


def test_build_cases_skips_unknown_target_and_prefix_papers():
    catalog = {"paper-1", "paper-2"}
    cases, stats = build_cases({"0": [1, 999], "1": [1, 999, 2]}, catalog)
    # user 0: held-out paper-999 unknown -> ineligible; user 1: 999 dropped from prefix.
    assert len(cases) == 1
    assert cases[0].prefix == [1]
    assert cases[0].target == 2
    assert stats.unknown_paper_skips == 2
    assert stats.ineligible_users == 1


def test_hit_and_reciprocal_rank_on_hand_built_rankings():
    ranked = [f"paper-{i}" for i in range(1, 21)]  # paper-11 at rank 11
    assert is_hit(ranked, "paper-1", 10) is True
    assert reciprocal_rank(ranked, "paper-1") == 1.0
    assert is_hit(ranked, "paper-10", 10) is True
    assert reciprocal_rank(ranked, "paper-10") == pytest.approx(0.1)
    assert is_hit(ranked, "paper-11", 10) is False
    assert reciprocal_rank(ranked, "paper-11") == pytest.approx(1 / 11)
    assert is_hit(ranked, "paper-99", 10) is False
    assert reciprocal_rank(ranked, "paper-99") == 0.0


def test_popularity_counts_never_include_held_out_targets():
    # user 0 holds out paper-8, user 1 holds out paper-9: only paper-7 may count.
    sequences = {"0": [7, 8], "1": [7, 9], "2": [5]}
    counts = _borrow_counts(sequences, {f"paper-{i}" for i in range(1, 10)})
    assert counts == {7: 2, 5: 1}
    ranking = popularity_ranking(counts)
    assert ranking == ["paper-7", "paper-5"]


def test_popularity_ranking_breaks_ties_by_paper_id():
    assert popularity_ranking({5: 2, 3: 2, 9: 1}) == ["paper-3", "paper-5", "paper-9"]


def _norm(*coords: float) -> list[float]:
    v = np.asarray(coords, dtype=np.float32)
    return list(v / np.linalg.norm(v))


def _tiny_docs() -> dict:
    return {
        f"paper-{i}": {
            "id": f"paper-{i}",
            "corpus": "catalog",
            "title": f"Title {i}",
            "metadata": {},
        }
        for i in range(1, 7)
    }


def _tiny_vectors() -> np.ndarray:
    # paper-2 ~ paper-1; for seed paper-3 the order is 5 > 6 > 4 > 2 > 1.
    return np.asarray(
        [
            [1.0, 0.0],  # paper-1
            _norm(1.0, 0.03),  # paper-2
            [0.0, 1.0],  # paper-3
            _norm(1.0, 1.0),  # paper-4
            _norm(0.2, 1.0),  # paper-5
            _norm(0.4, 1.0),  # paper-6
        ],
        dtype=np.float32,
    )


def test_evaluate_end_to_end_exact_metrics():
    docs = _tiny_docs()
    vectors = _tiny_vectors()
    sequences = {"0": [1, 2], "1": [3, 4]}
    titles = {"Title 1": ["paper-5", "paper-2"], "Title 3": ["paper-4", "paper-1"]}

    outcome = evaluate(docs, vectors, sequences, lambda q: titles[q], hit_at=10)

    assert outcome["stats"].eligible_users == 2
    results = outcome["results"]
    # recommend: paper-2 at rank 1, paper-4 at rank 3.
    assert results["recommend"]["hit_at_k"] == 1.0
    assert results["recommend"]["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    # popularity: only prefix papers count; each user's list excludes their
    # own prefix, so neither target is ever suggested.
    assert results["popularity"]["hit_at_k"] == 0.0
    assert results["popularity"]["mrr"] == 0.0
    # title-as-query: paper-2 at rank 2, paper-4 at rank 1.
    assert results["title-as-query"]["hit_at_k"] == 1.0
    assert results["title-as-query"]["mrr"] == pytest.approx((0.5 + 1.0) / 2)


def _outcome_with_users(n: int) -> dict:
    docs = {
        f"paper-{i}": {"id": f"paper-{i}", "corpus": "catalog", "title": f"T{i}", "metadata": {}}
        for i in range(1, 2 * n + 1)
    }
    vectors = np.eye(2 * n, dtype=np.float32)
    sequences = {str(u): [2 * u + 1, 2 * u + 2] for u in range(n)}
    return evaluate(docs, vectors, sequences, lambda q: [], hit_at=10)


def test_report_labels_thin_data_informational():
    report = format_report(_outcome_with_users(19), "x.json")
    assert "INFORMATIONAL ONLY: fewer than 20 eligible users (19)" in report
    report = format_report(_outcome_with_users(20), "x.json")
    assert "INFORMATIONAL ONLY" not in report


def test_report_always_shows_user_count_and_approaches():
    report = format_report(_outcome_with_users(3), "x.json")
    assert "Users in file                         : 3" in report
    assert "Eligible users (>=2 distinct borrows) : 3" in report
    for name in ("recommend", "popularity", "title-as-query"):
        assert name in report
