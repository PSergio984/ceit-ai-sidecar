"""Postgres index store: documents + generated tsv + single-row index_state.

Neon/Lakebase-ready: connections disable prepared statements (pooled
endpoints run transaction pooling) and names are schema-qualified via
psycopg.sql.Identifier — never the search_path startup parameter poolers
reject. sync() is incremental: content-hash diff, embed only added/changed,
one transaction (advisory-xact-locked) so readers see the old index until
commit, then the keyword index + VACUUM on an autocommit connection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .ingest import embed_documents, load_documents, source_generated_at
from .rebuild import CONTRACT_VERSION

logger = logging.getLogger("ceit.sidecar")

# Serialises concurrent syncs across processes; transaction-scoped so it is
# safe under transaction pooling.
_SYNC_LOCK_KEY = 727_272_006
_UPSERT_BATCH = 200


def _content_hash(model_name: str, doc: dict) -> str:
    """sha256 over canonical JSON — includes model_name so a model change
    re-embeds every document."""
    payload = {
        "model_name": model_name,
        "id": doc["id"],
        "corpus": doc["corpus"],
        "title": doc["title"],
        "text": doc["text"],
        "metadata": doc["metadata"],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _vector_literal(vector) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgIndexStore:
    """IndexStore backed by Postgres (documents + index_state tables)."""

    name = "pgvector"

    def __init__(self, database_url: str, schema: str = "public"):
        self.database_url = database_url
        self.schema = schema
        self._docs = sql.Identifier(schema, "documents")
        self._state = sql.Identifier(schema, "index_state")
        # CREATE INDEX rejects schema-qualified index names — the index lands
        # in the table's schema automatically.
        self._corpus_idx = sql.Identifier("documents_corpus_idx")
        self._bm25_idx = sql.Identifier("documents_bm25")
        self._gin_idx = sql.Identifier("documents_tsv_gin")

    def _connect(self, *, autocommit: bool = False) -> psycopg.Connection:
        # prepare_threshold=None: pooled endpoints cannot serve prepared
        # statements across pooled transactions.
        return psycopg.connect(self.database_url, autocommit=autocommit, prepare_threshold=None)

    def _lakebase_available(self, conn) -> bool:
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'lakebase_text')"
        ).fetchone()[0]

    def _index_exists(self, conn, index_name: str) -> bool:
        return conn.execute(
            "SELECT to_regclass(%s) IS NOT NULL", (f"{self.schema}.{index_name}",)
        ).fetchone()[0]

    def _ensure_schema(self, conn) -> None:
        if not conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
        ).fetchone()[0]:
            raise RuntimeError(
                "pgvector extension is not available on this database"
                " (install pgvector or use a pgvector-capable service)"
            )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if self._lakebase_available(conn):
            conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_text")
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
        conn.execute(
            sql.SQL(
                """CREATE TABLE IF NOT EXISTS {} (
                id text PRIMARY KEY,
                corpus text NOT NULL,
                title text NOT NULL,
                text text NOT NULL,
                metadata jsonb NOT NULL,
                content_hash text NOT NULL,
                embedding vector(384) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('simple', regexp_replace(text, '[^[:alnum:]]+', ' ', 'g')))
                    STORED
            )"""
            ).format(self._docs)
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (corpus)").format(
                self._corpus_idx, self._docs
            )
        )
        conn.execute(
            sql.SQL(
                """CREATE TABLE IF NOT EXISTS {} (
                id smallint PRIMARY KEY CHECK (id = 1),
                version int NOT NULL,
                built_at timestamptz NOT NULL,
                source_generated_at text,
                documents int NOT NULL,
                embedded int NOT NULL,
                by_corpus jsonb NOT NULL,
                model_name text NOT NULL,
                contract_version text NOT NULL,
                keyword_index text
            )"""
            ).format(self._state)
        )

    def load_state(self) -> dict | None:
        with self._connect(autocommit=True) as conn:
            conn.execute("SELECT 1")  # wakes a sleeping Neon compute
            try:
                row = conn.execute(
                    sql.SQL(
                        "SELECT version, built_at, source_generated_at, documents,"
                        " embedded, by_corpus, model_name, contract_version,"
                        " keyword_index FROM {} WHERE id = 1"
                    ).format(self._state)
                ).fetchone()
            except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName):
                return None
        if row is None:
            return None
        return {
            "version": row[0],
            "built_at": row[1].isoformat() if row[1] else None,
            "source_generated_at": row[2],
            "documents": row[3],
            "embedded": row[4],
            "by_corpus": row[5],
            "model_name": row[6],
            "contract_version": row[7],
            "keyword_index": row[8],
        }

    def sync(self, corpus_path: Path, model_name: str, embed_fn=None) -> dict:
        """Incrementally sync the index to the corpus; returns the new state.

        load_documents() validates BEFORE any connection so a bad corpus
        never touches the database. Everything else is one transaction —
        any failure rolls back, leaving the previous index intact.
        """
        docs = load_documents(corpus_path)
        source_generated = source_generated_at(corpus_path)

        with self._connect() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SYNC_LOCK_KEY,))
            self._ensure_schema(conn)
            lakebase = self._lakebase_available(conn)
            keyword_index = "lakebase_bm25" if lakebase else "gin"

            stored = dict(
                conn.execute(
                    sql.SQL("SELECT id, content_hash FROM {}").format(self._docs)
                ).fetchall()
            )
            wanted = {d["id"]: _content_hash(model_name, d) for d in docs}
            added = [d for d in docs if d["id"] not in stored]
            changed = [d for d in docs if d["id"] in stored and stored[d["id"]] != wanted[d["id"]]]
            removed = [doc_id for doc_id in stored if doc_id not in wanted]
            dirty = bool(added or changed or removed)

            to_embed = added + changed
            if to_embed:
                vectors = (
                    embed_fn(to_embed)
                    if embed_fn is not None
                    else embed_documents(to_embed, model_name)
                )
                upsert = sql.SQL(
                    """INSERT INTO {}
                        (id, corpus, title, text, metadata, content_hash, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (id) DO UPDATE SET
                            corpus = EXCLUDED.corpus,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            content_hash = EXCLUDED.content_hash,
                            embedding = EXCLUDED.embedding"""
                ).format(self._docs)
                rows = [
                    (
                        d["id"],
                        d["corpus"],
                        d["title"],
                        d["text"],
                        Jsonb(d["metadata"]),
                        wanted[d["id"]],
                        _vector_literal(v),
                    )
                    for d, v in zip(to_embed, vectors)
                ]
                with conn.cursor() as cur:
                    for start in range(0, len(rows), _UPSERT_BATCH):
                        cur.executemany(upsert, rows[start : start + _UPSERT_BATCH])
            if removed:
                conn.execute(
                    sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(self._docs),
                    (removed,),
                )

            prev = conn.execute(
                sql.SQL("SELECT version, built_at FROM {} WHERE id = 1").format(self._state)
            ).fetchone()
            if dirty or prev is None:
                version = (prev[0] if prev else 0) + 1
                built_at = datetime.now(UTC)
            else:
                version, built_at = prev
            by_corpus = dict(Counter(d["corpus"] for d in docs))
            conn.execute(
                sql.SQL(
                    """INSERT INTO {}
                        (id, version, built_at, source_generated_at, documents,
                         embedded, by_corpus, model_name, contract_version,
                         keyword_index)
                        VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            version = EXCLUDED.version,
                            built_at = EXCLUDED.built_at,
                            source_generated_at = EXCLUDED.source_generated_at,
                            documents = EXCLUDED.documents,
                            embedded = EXCLUDED.embedded,
                            by_corpus = EXCLUDED.by_corpus,
                            model_name = EXCLUDED.model_name,
                            contract_version = EXCLUDED.contract_version,
                            keyword_index = EXCLUDED.keyword_index"""
                ).format(self._state),
                (
                    version,
                    built_at,
                    source_generated,
                    len(docs),
                    len(docs),
                    Jsonb(by_corpus),
                    model_name,
                    CONTRACT_VERSION,
                    keyword_index,
                ),
            )

        # After the commit: the keyword index must see a populated table
        # (spike finding) and VACUUM cannot run inside a transaction.
        if dirty:
            self._post_commit_maintenance(lakebase)

        logger.info(
            "pg sync: added=%d changed=%d removed=%d unchanged=%d",
            len(added),
            len(changed),
            len(removed),
            len(docs) - len(added) - len(changed),
        )
        return {
            "version": version,
            "built_at": built_at.isoformat(),
            "source_generated_at": source_generated,
            "documents": len(docs),
            "embedded": len(docs),
            "by_corpus": by_corpus,
            "model_name": model_name,
            "contract_version": CONTRACT_VERSION,
            "keyword_index": keyword_index,
        }

    def _post_commit_maintenance(self, lakebase: bool) -> None:
        with self._connect(autocommit=True) as conn:
            if lakebase:
                if not self._index_exists(conn, "documents_bm25"):
                    conn.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {} ON {}"
                            " USING lakebase_bm25 (tsv) WITH (k1 = 1.2, b = 0.75)"
                        ).format(self._bm25_idx, self._docs)
                    )
            elif not self._index_exists(conn, "documents_tsv_gin"):
                conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (tsv)").format(
                        self._gin_idx, self._docs
                    )
                )
            conn.execute(sql.SQL("VACUUM (ANALYZE) {}").format(self._docs))
