"""Ingest validation: envelope checks, corpus tags, duplicate ids."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.ingest import load_documents
from tests.conftest import make_corpus


def test_loads_catalog_and_policy_docs_with_corpus_tags(tmp_path):
    corpus = make_corpus(tmp_path)
    docs = load_documents(corpus)

    corpora = {d["corpus"] for d in docs}
    assert corpora == {"catalog", "policy"}
    assert len(docs) == 6


def test_duplicate_id_raises(tmp_path):
    corpus = make_corpus(tmp_path)
    payload = json.loads((corpus / "catalog.json").read_text(encoding="utf-8"))
    payload["documents"].append(dict(payload["documents"][0]))  # duplicate id paper-1
    (corpus / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document id"):
        load_documents(corpus)


def test_missing_generated_at_raises(tmp_path):
    corpus = make_corpus(tmp_path)
    payload = json.loads((corpus / "catalog.json").read_text(encoding="utf-8"))
    del payload["generated_at"]
    (corpus / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generated_at"):
        load_documents(corpus)


def test_wrong_schema_version_raises(tmp_path):
    corpus = make_corpus(tmp_path)
    payload = json.loads((corpus / "catalog.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    (corpus / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_documents(corpus)


def test_missing_corpus_files_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no corpus files found"):
        load_documents(empty)


def test_embedding_produces_one_vector_per_document():
    """Whole-document embedding: one vector per doc, no chunking (R6)."""
    try:
        from fastembed import TextEmbedding
    except (ImportError, OSError):  # pragma: no cover - offline env
        pytest.skip("fastembed not installed")

    try:
        model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except (OSError, ValueError):  # pragma: no cover - no model cache / no network
        pytest.skip("model not cached and network unavailable")

    texts = [
        "find a thesis about water pumps",
        "Design of a Smart Flood Monitoring System",
    ]
    vectors = np.asarray(list(model.embed(texts)), dtype=np.float32)
    assert vectors.shape == (2, 384)
