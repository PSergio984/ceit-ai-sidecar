"""Offline embed-and-push script (#8): LangChain PGVectorStore parity tests.

Requires TEST_DATABASE_URL (CI provides a pgvector container) and the optional
`offline` uv group (langchain-postgres). Each test uses fresh ceit_test_<hex>
schemas dropped on teardown — public is never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

import numpy as np
import psycopg
import pytest
from conftest import make_corpus
from psycopg import sql

pytest.importorskip("langchain_postgres", reason="offline dependency group not installed")

from app.pg_index import PgIndexStore
from scripts.push_embeddings import push

PG_URL = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parent.parent


def _ident(schema: str, table: str):
    return sql.Identifier(schema, table)


class CountingEmbedder:
    """Deterministic 384-dim embedder recording each batch's doc count."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def __call__(self, docs) -> np.ndarray:
        self.batch_sizes.append(len(docs))
        vectors = []
        for doc in docs:
            seed = int.from_bytes(hashlib.sha256(doc["text"].encode()).digest()[:4])
            v = np.random.RandomState(seed).rand(384).astype(np.float32)
            vectors.append(v / np.linalg.norm(v))
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def schema_factory():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    created: list[str] = []

    def make() -> str:
        schema = f"ceit_test_{secrets.token_hex(4)}"
        with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        created.append(schema)
        return schema

    yield make
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        for schema in created:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _push(corpus_path: Path, schema: str, embedder, force_full: bool = False) -> dict:
    return push(
        corpus_path,
        database_url=PG_URL,
        schema=schema,
        model_name="test-model",
        force_full=force_full,
        embed_fn=embedder,
    )


def _read_rows(schema: str) -> dict[str, tuple]:
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        rows = conn.execute(
            sql.SQL(
                "SELECT id, corpus, title, metadata, content_hash, embedding::text"
                " FROM {} ORDER BY id"
            ).format(_ident(schema, "documents"))
        ).fetchall()
    return {row[0]: row[1:] for row in rows}


def _vec(raw) -> np.ndarray:
    return np.asarray(json.loads(raw), dtype=np.float32)


def _state(schema: str) -> dict:
    return PgIndexStore(PG_URL, schema=schema).load_state()


def test_script_and_pg_sync_produce_identical_index(schema_factory, tmp_path):
    """Parity: the LangChain path and PgIndexStore.sync must produce the same
    ids, content hashes and embeddings on the same corpus."""
    corpus = make_corpus(tmp_path)
    script_schema, sync_schema = schema_factory(), schema_factory()
    script_embedder, sync_embedder = CountingEmbedder(), CountingEmbedder()

    script_state = _push(corpus, script_schema, script_embedder)
    sync_state = PgIndexStore(PG_URL, schema=sync_schema).sync(
        corpus, "test-model", embed_fn=sync_embedder
    )

    script_rows, sync_rows = _read_rows(script_schema), _read_rows(sync_schema)
    assert set(script_rows) == set(sync_rows)
    assert {row[3] for row in script_rows.values()} == {
        row[3] for row in sync_rows.values()
    }  # identical content_hash sets
    for doc_id, srow in script_rows.items():
        prow = sync_rows[doc_id]
        assert srow[:4] == prow[:4]  # corpus, title, metadata, content_hash
        assert np.allclose(_vec(srow[4]), _vec(prow[4]))
    assert script_state["version"] == sync_state["version"]
    assert script_embedder.batch_sizes == sync_embedder.batch_sizes


def test_second_run_is_complete_noop(schema_factory, tmp_path):
    """An unchanged corpus push must not call the embedder or bump the version."""
    corpus = make_corpus(tmp_path)
    schema = schema_factory()
    embedder = CountingEmbedder()
    first = _push(corpus, schema, embedder)
    rows_before = _read_rows(schema)

    embedder.batch_sizes.clear()
    second = _push(corpus, schema, embedder)

    assert embedder.batch_sizes == []  # embedder never invoked
    assert second["version"] == first["version"]
    assert _state(schema)["version"] == first["version"]
    assert _read_rows(schema) == rows_before


def test_force_full_reembeds_everything(schema_factory, tmp_path):
    corpus = make_corpus(tmp_path)
    schema = schema_factory()
    embedder = CountingEmbedder()
    first = _push(corpus, schema, embedder)
    total = _state(schema)["documents"]

    embedder.batch_sizes.clear()
    state = _push(corpus, schema, embedder, force_full=True)

    assert embedder.batch_sizes == [total]
    assert state["version"] == first["version"] + 1


def test_dirty_diff_embeds_only_changed_rows(schema_factory, tmp_path):
    """Changed + added docs are re-embedded; removed ids are deleted; the rest
    of the index is untouched."""
    corpus = make_corpus(tmp_path)
    schema = schema_factory()
    embedder = CountingEmbedder()
    first = _push(corpus, schema, embedder)

    catalog_path = corpus / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][0]["text"] += " offline change"
    new_doc = dict(catalog["documents"][2])
    new_doc["id"] = "paper-offline-new"
    catalog["documents"].append(new_doc)
    removed_id = catalog["documents"].pop(1)["id"]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    embedder.batch_sizes.clear()
    second = _push(corpus, schema, embedder)

    assert embedder.batch_sizes == [2]  # only the changed + added docs
    assert second["version"] == first["version"] + 1
    rows = _read_rows(schema)
    assert removed_id not in rows
    assert "paper-offline-new" in rows
    assert len(rows) == 6


def test_embedder_failure_leaves_previous_index(schema_factory, tmp_path):
    corpus = make_corpus(tmp_path)
    schema = schema_factory()
    first = _push(corpus, schema, CountingEmbedder())

    catalog_path = corpus / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][0]["text"] += " changed"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    def boom(docs):
        raise RuntimeError("embedder exploded")

    with pytest.raises(RuntimeError, match="embedder exploded"):
        _push(corpus, schema, boom)

    assert _state(schema)["version"] == first["version"]
    assert len(_read_rows(schema)) == 6


def test_lc_writes_roll_back_when_state_write_fails(schema_factory, tmp_path, monkeypatch):
    """Atomicity proof: the LangChain upserts share one transaction with the
    diff/delete/state writes — a failure after the upserts rolls them back."""
    corpus = make_corpus(tmp_path)
    schema = schema_factory()
    _push(corpus, schema, CountingEmbedder())
    rows_before = _read_rows(schema)

    catalog_path = corpus / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    doomed = dict(catalog["documents"][1])
    doomed["id"] = "paper-doomed"
    catalog["documents"].append(doomed)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    import scripts.push_embeddings as mod

    def explode(*args, **kwargs):
        raise RuntimeError("state write exploded")

    monkeypatch.setattr(mod, "_write_index_state", explode)
    with pytest.raises(RuntimeError, match="state write exploded"):
        _push(corpus, schema, CountingEmbedder())

    assert _read_rows(schema) == rows_before  # LC upsert rolled back too
