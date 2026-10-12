"""Pure-function fusion tests: deterministic ordering, filters, pin, shaping."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.fusion import fuse, passes_filters, pin_catalog_code, shape_results

REPO = Path(__file__).resolve().parent.parent


def _doc(doc_id: str, corpus: str = "catalog", **meta) -> dict:
    return {
        "id": doc_id,
        "corpus": corpus,
        "title": f"Title {doc_id}",
        "text": f"text of {doc_id}",
        "metadata": meta,
    }


def test_hybrid_ties_break_by_id_regardless_of_input_order():
    # paper-a: bm25 rank 2 + semantic rank 1; paper-b: bm25 rank 1 + semantic
    # rank 2 — identical fused scores, so the id decides.
    bm25 = {"paper-b": 1, "paper-a": 2}
    sem = {"paper-a": 0.9, "paper-b": 0.8}
    ranked, scores, _ = fuse("hybrid", 60, bm25, sem)
    assert scores["paper-a"] == scores["paper-b"]
    assert ranked == ["paper-a", "paper-b"]

    ranked2, _, _ = fuse(
        "hybrid", 60, {"paper-a": 2, "paper-b": 1}, {"paper-b": 0.8, "paper-a": 0.9}
    )
    assert ranked2 == ranked


def test_fuse_bm25_and_semantic_orderings():
    ranked, scores, sem_map = fuse("bm25", 60, {"a": 2, "b": 1}, {})
    assert ranked == ["b", "a"]
    assert scores["b"] == pytest.approx(1 / 61)
    assert sem_map == {}

    ranked, scores, sem_map = fuse("semantic", 60, {}, {"b": 0.5, "a": 0.9, "c": 0.9})
    # a and c tie on cosine -> id order; both outrank b.
    assert ranked == ["a", "c", "b"]
    assert scores["b"] == pytest.approx(0.5)
    assert sem_map == {"a": 1, "c": 2, "b": 3}


def test_passes_filters():
    doc = _doc(
        "p",
        paper_type="Thesis",
        department="Civil Engineering",
        publication_year=2015,
        authors=["Juan Dela Cruz"],
        research_adviser="Engr. Jose Santos",
        technical_adviser="",
    )

    assert passes_filters(doc, {"paper_type": "Thesis"}, None)
    assert not passes_filters(doc, {"paper_type": "Capstone"}, None)
    assert passes_filters(doc, {"department": "Civil Engineering"}, None)
    assert not passes_filters(doc, {"department": "Electrical Engineering"}, None)
    assert passes_filters(doc, {"publication_year": 2015}, None)
    assert not passes_filters(doc, {"publication_year": "2016"}, None)
    assert passes_filters(doc, {"year_from": "2010", "year_to": "2020"}, None)
    assert not passes_filters(doc, {"year_from": "2016"}, None)
    assert not passes_filters(doc, {"year_to": "2014"}, None)
    assert passes_filters(doc, {"author": "dela cruz"}, None)
    assert not passes_filters(doc, {"author": "santos"}, None)
    assert passes_filters(doc, {"adviser": "jose santos"}, None)
    assert not passes_filters(doc, {"adviser": "reyes"}, None)
    assert passes_filters(doc, {}, "catalog")
    assert not passes_filters(doc, {}, "policy")
    assert passes_filters(doc, None, None)


def test_pin_catalog_code():
    docs = {
        "a": _doc("a", catalog_code="CEIT-CE-15-014"),
        "b": _doc("b", catalog_code="CEIT-EE-25-01"),
    }
    ranked = ["a", "b"]

    assert pin_catalog_code("ceit-ee-25-01", ranked, docs) == "b"
    assert ranked == ["b", "a"]

    assert pin_catalog_code("water pumps", ["a", "b"], docs) is None
    assert pin_catalog_code("CEIT-XX-99-00", ["a", "b"], docs) is None


def test_shape_results_fields_rounding_and_ranks():
    docs = {
        "a": _doc("a", catalog_code="CEIT-EE-25-01"),
        "b": _doc("b"),
    }
    scores = {"a": 0.123456789, "b": 0.01}
    bm25 = {"a": 1, "b": 2}
    sem_map = {"b": 1, "a": 2}

    results = shape_results(["a", "b"], 10, docs, scores, bm25, sem_map, "hybrid", "a", True)
    assert results[0] == {
        "id": "a",
        "corpus": "catalog",
        "title": "Title a",
        "score": round(0.123456789, 4),
        "bm25_rank": 1,
        "semantic_rank": 2,
        "pinned": True,
        "metadata": {"catalog_code": "CEIT-EE-25-01"},
        "text": "text of a",
    }
    assert results[1]["pinned"] is False

    no_text = shape_results(["a"], 10, docs, scores, bm25, sem_map, "hybrid", None, False)
    assert "text" not in no_text[0]

    bm25_only = shape_results(["a"], 10, docs, scores, bm25, {}, "bm25", None, False)
    assert bm25_only[0]["bm25_rank"] == 1
    assert bm25_only[0]["semantic_rank"] is None

    semantic_only = shape_results(["a"], 10, docs, scores, {}, sem_map, "semantic", None, False)
    assert semantic_only[0]["bm25_rank"] is None
    assert semantic_only[0]["semantic_rank"] == 2

    capped = shape_results(["a", "b"], 1, docs, scores, bm25, sem_map, "hybrid", None, False)
    assert [r["id"] for r in capped] == ["a"]


def _hybrid_order_with_seed(seed: str, workdir: Path) -> list[str]:
    """Run rrf_search in a subprocess under a fixed PYTHONHASHSEED."""
    workdir.mkdir(parents=True)
    script = f"""
import json, sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(REPO / "tests")!r})
from pathlib import Path
from conftest import build_test_index, make_corpus
from app.search import HybridSearch

workdir = Path({str(workdir)!r})
corpus = make_corpus(workdir)
cache, _docs = build_test_index(workdir, corpus)
hs = HybridSearch(cache, "m")
# Genuine fused-score tie: paper-1 = bm25 rank 2 + semantic rank 1,
# paper-2 = bm25 rank 1 + semantic rank 2.
hs._bm25_ranks = lambda db, q, c: {{"paper-2": 1, "paper-1": 2}}
hs._semantic_scores = lambda d, v, q: {{"paper-1": 0.9, "paper-2": 0.8}}
print(json.dumps([r["id"] for r in hs.rrf_search("q", limit=10)]))
"""
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO), "SIDECAR_TOKEN": "test"}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=workdir,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fused_tie_order_is_stable_across_hash_seeds(tmp_path):
    order1 = _hybrid_order_with_seed("1", tmp_path / "run1")
    order2 = _hybrid_order_with_seed("2", tmp_path / "run2")
    assert order1 == order2 == ["paper-1", "paper-2"]
