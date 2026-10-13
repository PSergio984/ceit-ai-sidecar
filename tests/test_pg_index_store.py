"""PgIndexStore: Postgres index storage + incremental sync + health (#6).

DB tests run only when TEST_DATABASE_URL is set (CI provides a pgvector
container) and each uses a fresh ceit_test_<hex> schema dropped on teardown
— public is never touched. Unit tests (hash, factory, sqlite health shape)
run without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
from pathlib import Path

import numpy as np
import psycopg
import pytest
from conftest import build_test_index, reset_main_singletons
from fastapi.testclient import TestClient
from psycopg import sql

from app.config import Settings
from app.index_store import SqliteIndexStore, create_index_store
from app.pg_index import PgIndexStore, _content_hash

PG_URL = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
H = {"X-Sidecar-Token": "test-token"}
BAD_URL = "postgresql://postgres:bad@127.0.0.1:59999/nope"


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
def pg_schema():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"ceit_test_{secrets.token_hex(4)}"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    yield schema
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def store(pg_schema):
    return PgIndexStore(PG_URL, schema=pg_schema)


def _client(
    corpus_dir: Path, schema: str, embedder, database_url: str, connect_timeout: int = 15
) -> TestClient:
    settings = Settings(
        sidecar_token="test-token",
        corpus_path=corpus_dir,
        model_name="test-model",
        vector_backend="pgvector",
        database_url=database_url,
        pg_index_schema=schema,
        pg_connect_timeout=connect_timeout,
    )
    import app.main as main_mod
    import app.rebuild as rebuild_mod

    main_mod.settings = settings
    reset_main_singletons(main_mod)
    main_mod._metrics = main_mod._fresh_metrics()
    rebuild_mod._embed_override = embedder
    return TestClient(main_mod.app)


def _doc_count(schema: str) -> int:
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        return conn.execute(
            sql.SQL("SELECT count(*) FROM {}").format(_ident(schema, "documents"))
        ).fetchone()[0]


# --- unit tests (no database) ------------------------------------------------


def test_content_hash_is_stable_and_field_sensitive():
    doc = {
        "id": "paper-1",
        "corpus": "catalog",
        "title": "T",
        "text": "body",
        "metadata": {"a": 1, "b": [2, 3]},
    }
    assert _content_hash("m", doc) == _content_hash("m", dict(doc))
    for field in ("id", "corpus", "title", "text"):
        changed = dict(doc, **{field: doc[field] + "x"})
        assert _content_hash("m", changed) != _content_hash("m", doc)
    changed_meta = dict(doc, metadata={"a": 2, "b": [2, 3]})
    assert _content_hash("m", changed_meta) != _content_hash("m", doc)
    assert _content_hash("other-model", doc) != _content_hash("m", doc)


def test_create_index_store_selects_sqlite_by_default():
    store = create_index_store(Settings(sidecar_token="t"))
    assert isinstance(store, SqliteIndexStore)
    assert store.name == "sqlite"


def test_create_index_store_pgvector_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        # database_url="" explicitly — Settings also reads .env, where a real
        # DATABASE_URL (e.g. a Neon link) would silently satisfy the check.
        create_index_store(Settings(sidecar_token="t", vector_backend="pgvector", database_url=""))
    store = create_index_store(
        Settings(sidecar_token="t", vector_backend="pgvector", database_url=BAD_URL)
    )
    assert isinstance(store, PgIndexStore)


def test_health_reports_sqlite_backend(tmp_path, corpus_path):
    cache, _ = build_test_index(tmp_path, corpus_path)
    import app.main as main_mod

    main_mod.settings = Settings(
        sidecar_token="test-token", corpus_path=corpus_path, cache_dir=str(cache)
    )
    reset_main_singletons(main_mod)
    main_mod._metrics = main_mod._fresh_metrics()
    body = TestClient(main_mod.app).get("/health", headers=H).json()
    assert body["status"] == "ok"
    assert body["backend"] == "sqlite"
    assert body["index"]["documents"] == 6


def test_health_degraded_when_pgvector_unconfigured(tmp_path):
    # No database needed: pgvector with an empty DATABASE_URL cannot even
    # construct the store — /health must still answer 200 degraded.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    app = _client(corpus_dir, "public", None, "")
    resp = app.get("/health", headers=H)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["index"] is None
    assert body["backend"] == "pgvector"


# --- Postgres tests (skipped without TEST_DATABASE_URL) -----------------------


def test_full_corpus_loads_and_health_reports_pgvector(pg_schema):
    app = _client(CORPUS, pg_schema, CountingEmbedder(), PG_URL)
    resp = app.post("/index/rebuild", headers=H)
    assert resp.status_code == 200
    assert resp.json()["documents"] == 1338

    body = app.get("/health", headers=H).json()
    assert body["status"] == "ok"
    assert body["backend"] == "pgvector"
    assert body["index"]["documents"] == 1338
    assert body["index"]["embedded"] == 1338
    assert body["index"]["by_corpus"] == {"catalog": 959, "policy": 379}
    assert _doc_count(pg_schema) == 1338


def test_unchanged_sync_embeds_nothing_and_keeps_version(store):
    embedder = CountingEmbedder()
    first = store.sync(CORPUS, "test-model", embed_fn=embedder)
    embedder.batch_sizes.clear()

    second = store.sync(CORPUS, "test-model", embed_fn=embedder)
    assert embedder.batch_sizes == []
    assert second["version"] == first["version"]
    assert second["built_at"] == first["built_at"]
    assert store.load_state()["version"] == first["version"]


def test_incremental_sync_embeds_only_changes(store, pg_schema, tmp_path):
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_dir)
    embedder = CountingEmbedder()
    first = store.sync(corpus_dir, "test-model", embed_fn=embedder)

    catalog_path = corpus_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][0]["text"] += " incremental change"
    new_doc = dict(catalog["documents"][2])
    new_doc["id"] = "paper-new-1"
    catalog["documents"].append(new_doc)
    removed_id = catalog["documents"].pop(1)["id"]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    embedder.batch_sizes.clear()
    second = store.sync(corpus_dir, "test-model", embed_fn=embedder)

    assert embedder.batch_sizes == [2]
    assert second["version"] == first["version"] + 1
    assert second["documents"] == first["documents"]
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        docs_tbl = _ident(pg_schema, "documents")
        gone = conn.execute(
            sql.SQL("SELECT count(*) FROM {} WHERE id = %s").format(docs_tbl),
            (removed_id,),
        ).fetchone()[0]
        present = conn.execute(
            sql.SQL("SELECT count(*) FROM {} WHERE id = %s").format(docs_tbl),
            ("paper-new-1",),
        ).fetchone()[0]
    assert gone == 0
    assert present == 1


def test_model_name_change_reembeds_everything(store):
    embedder = CountingEmbedder()
    store.sync(CORPUS, "model-a", embed_fn=embedder)
    embedder.batch_sizes.clear()
    store.sync(CORPUS, "model-b", embed_fn=embedder)
    assert embedder.batch_sizes == [1338]


def test_concurrent_reader_never_sees_partial_index(store, pg_schema, tmp_path):
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus_dir)
    store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())
    old_count = 1338

    catalog_path = corpus_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for i in range(2):
        extra = dict(catalog["documents"][i])
        extra["id"] = f"paper-new-{i}"
        catalog["documents"].append(extra)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    new_count = old_count + 2

    blocking = threading.Event()
    release = threading.Event()

    def blocking_embed(docs):
        blocking.set()
        release.wait(timeout=60)
        return CountingEmbedder()(docs)

    sync_thread = threading.Thread(
        target=store.sync,
        args=(corpus_dir, "test-model"),
        kwargs={"embed_fn": blocking_embed},
    )
    sync_thread.start()
    assert blocking.wait(timeout=60)

    poll = sql.SQL("SELECT (SELECT documents FROM {}), (SELECT count(*) FROM {})").format(
        _ident(pg_schema, "index_state"), _ident(pg_schema, "documents")
    )
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        # While the writer transaction is open but uncommitted, every
        # single-statement snapshot must show the OLD state, consistently.
        for _ in range(5):
            state_docs, row_count = conn.execute(poll).fetchone()
            assert state_docs == row_count == old_count
        release.set()
        sync_thread.join(timeout=60)
        state_docs, row_count = conn.execute(poll).fetchone()
        assert (state_docs, row_count) == (new_count, new_count)
    assert not sync_thread.is_alive()


def test_invalid_upload_preserves_previous_index(store, pg_schema, tmp_path):
    store.sync(CORPUS, "test-model", embed_fn=CountingEmbedder())
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    app = _client(corpus_dir, pg_schema, CountingEmbedder(), PG_URL)

    bad = json.dumps({"schema_version": 2, "documents": []}).encode()
    resp = app.post(
        "/corpus/upload",
        files={"catalog": ("catalog.json", bad, "application/json")},
        headers=H,
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "upload_failed"
    assert _doc_count(pg_schema) == 1338
    assert app.get("/health", headers=H).json()["index"]["documents"] == 1338


def test_failing_embedder_rolls_back(store, pg_schema):
    first = store.sync(CORPUS, "model-a", embed_fn=CountingEmbedder())

    def boom(docs):
        raise RuntimeError("embedder exploded")

    with pytest.raises(RuntimeError, match="embedder exploded"):
        # model change forces every doc into the added/changed set
        store.sync(CORPUS, "model-b", embed_fn=boom)

    assert _doc_count(pg_schema) == 1338
    state = store.load_state()
    assert state["version"] == first["version"]
    assert state["model_name"] == "model-a"


def test_health_degraded_when_database_unreachable(pg_schema, tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    app = _client(corpus_dir, "public", None, BAD_URL, connect_timeout=1)
    resp = app.get("/health", headers=H)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["index"] is None
    assert body["backend"] == "pgvector"


def test_keyword_index_matches_available_extension(store, pg_schema):
    store.sync(CORPUS, "test-model", embed_fn=CountingEmbedder())
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        lakebase = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'lakebase_text')"
        ).fetchone()[0]
    expected_index = "documents_bm25" if lakebase else "documents_tsv_gin"
    expected_name = "lakebase_bm25" if lakebase else "gin"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        assert conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (f"{pg_schema}.{expected_index}",)
        ).fetchone()[0]
    assert store.load_state()["keyword_index"] == expected_name


def test_gin_fallback_when_lakebase_unavailable(store, pg_schema, monkeypatch):
    # On Neon lakebase_text exists; force the fallback to prove the branch.
    monkeypatch.setattr(store, "_lakebase_available", lambda conn: False)
    state = store.sync(CORPUS, "test-model", embed_fn=CountingEmbedder())
    assert state["keyword_index"] == "gin"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        assert conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (f"{pg_schema}.documents_tsv_gin",)
        ).fetchone()[0]


def _tiny_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    docs = [
        {
            "id": f"d{i}",
            "corpus": "catalog",
            "title": f"doc {i}",
            "text": f"text body {i}",
            "metadata": {},
        }
        for i in range(3)
    ]
    (corpus_dir / "catalog.json").write_text(
        json.dumps(
            {"schema_version": 1, "generated_at": "2026-01-01T00:00:00Z", "documents": docs}
        ),
        encoding="utf-8",
    )
    return corpus_dir


def test_maintenance_failure_after_commit_keeps_upload(pg_schema, tmp_path, monkeypatch):
    # Post-commit maintenance (CREATE INDEX / VACUUM) failing must not turn
    # an already-committed sync into an upload failure that deletes files.
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    app = _client(corpus_dir, pg_schema, CountingEmbedder(), PG_URL)

    def explode(self, lakebase, dirty):
        raise psycopg.OperationalError("simulated maintenance failure")

    monkeypatch.setattr(PgIndexStore, "_post_commit_maintenance", explode)
    catalog = json.dumps(
        {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00Z",
            "documents": [
                {
                    "id": "d1",
                    "corpus": "catalog",
                    "title": "one",
                    "text": "alpha",
                    "metadata": {},
                }
            ],
        }
    ).encode()
    resp = app.post(
        "/corpus/upload",
        files={"catalog": ("catalog.json", catalog, "application/json")},
        headers=H,
    )
    assert resp.status_code == 200
    assert (corpus_dir / "catalog.json").exists()  # committed -> not removed
    assert _doc_count(pg_schema) == 1


def test_unchanged_sync_recreates_missing_keyword_index(store, pg_schema, tmp_path):
    corpus_dir = _tiny_corpus(tmp_path)
    store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())
    index_name = (
        "documents_bm25"
        if store.load_state()["keyword_index"] == "lakebase_bm25"
        else "documents_tsv_gin"
    )
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(pg_schema, index_name)))

    embedder = CountingEmbedder()
    store.sync(corpus_dir, "test-model", embed_fn=embedder)
    assert embedder.batch_sizes == []  # unchanged corpus -> no embedder calls
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        assert conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (f"{pg_schema}.{index_name}",)
        ).fetchone()[0]


def test_unchanged_sync_checks_index_but_skips_vacuum(store, tmp_path, monkeypatch):
    corpus_dir = _tiny_corpus(tmp_path)
    store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())

    ensure_calls = []
    vacuum_calls = []
    monkeypatch.setattr(
        store,
        "_ensure_keyword_index",
        lambda conn, lakebase: ensure_calls.append(lakebase),
    )
    monkeypatch.setattr(store, "_vacuum", lambda conn: vacuum_calls.append(True))
    store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())
    assert len(ensure_calls) == 1
    assert vacuum_calls == []


def test_embedder_wrong_vector_count_rolls_back(store, pg_schema, tmp_path):
    corpus_dir = _tiny_corpus(tmp_path)
    first = store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())

    catalog_path = corpus_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][0]["text"] += " changed"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    def too_few(docs):
        return np.zeros((len(docs) - 1, 384), dtype=np.float32)

    def too_many(docs):
        return np.zeros((len(docs) + 1, 384), dtype=np.float32)

    with pytest.raises(ValueError, match="vectors for"):
        store.sync(corpus_dir, "test-model", embed_fn=too_few)
    with pytest.raises(ValueError, match="vectors for"):
        store.sync(corpus_dir, "test-model", embed_fn=too_many)

    assert _doc_count(pg_schema) == 3
    state = store.load_state()
    assert state["version"] == first["version"]
    assert state["model_name"] == "test-model"


def test_state_reports_live_table_counts(store, pg_schema, tmp_path):
    corpus_dir = _tiny_corpus(tmp_path)
    returned = store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())
    state = store.load_state()
    for key in ("documents", "embedded", "by_corpus"):
        assert returned[key] == state[key]

    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(
            sql.SQL("DELETE FROM {} WHERE id = %s").format(_ident(pg_schema, "documents")),
            ("d0",),
        )

    state = store.load_state()
    assert state["documents"] == 2
    assert state["embedded"] == 2
    assert sum(state["by_corpus"].values()) == 2
    assert state["version"] == returned["version"]


def test_load_state_never_reads_a_torn_index(store, pg_schema, tmp_path, monkeypatch):
    corpus_dir = _tiny_corpus(tmp_path)
    store.sync(corpus_dir, "test-model", embed_fn=CountingEmbedder())

    # A second sync adds a doc (bumps version to 2); commit it BETWEEN the
    # state-row read and the counts read by wrapping _live_counts.
    grown_dir = tmp_path / "grown"
    shutil.copytree(corpus_dir, grown_dir)
    cat_path = grown_dir / "catalog.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    extra = dict(cat["documents"][0])
    extra["id"] = "d-extra"
    cat["documents"].append(extra)
    cat_path.write_text(json.dumps(cat), encoding="utf-8")

    other = PgIndexStore(PG_URL, schema=pg_schema)
    original = PgIndexStore._live_counts

    def wrapped(conn):
        other.sync(grown_dir, "test-model", embed_fn=CountingEmbedder())
        return original(store, conn)

    monkeypatch.setattr(store, "_live_counts", wrapped)
    state = store.load_state()

    # consistent with ONE committed index: v1+3 or v2+4, never v1+4
    assert (state["version"], state["documents"]) in {(1, 3), (2, 4)}


def test_index_exists_check_handles_mixed_case_schema():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"CeitMixed_{secrets.token_hex(3)}"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            conn.execute(
                sql.SQL("CREATE TABLE {} (t tsvector)").format(sql.Identifier(schema, "documents"))
            )
            conn.execute(
                sql.SQL("CREATE INDEX {} ON {} USING gin (t)").format(
                    sql.Identifier("documents_tsv_gin"), sql.Identifier(schema, "documents")
                )
            )
            store = PgIndexStore(PG_URL, schema=schema)
            assert store._index_exists(conn, "documents_tsv_gin")
            assert not store._index_exists(conn, "documents_missing_idx")
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
