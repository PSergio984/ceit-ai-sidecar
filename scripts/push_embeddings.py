"""Offline embed-and-push (#8): FastEmbed -> LangChain PGVectorStore -> Postgres.

    uv run --group offline python scripts/push_embeddings.py <corpus> [--force-full]

<corpus> is the export directory holding catalog.json / policies.json (a file
inside it works too — its parent directory is used). Connection settings come
from the environment, same as the app: DATABASE_URL (required), PG_INDEX_SCHEMA
(default public), MODEL_NAME, PG_CONNECT_TIMEOUT.

Rows go through langchain_postgres' AsyncPGVectorStore (id_column=id,
content_column=text, embedding_column=embedding, metadata_json_column=metadata,
metadata_columns=[corpus, title, content_hash]) into the same
<schema>.documents table PgIndexStore.sync manages. Embeddings come from
app.ingest.embed_documents — the same model and normalization as the in-app
ingest — so either path leaves the index in the same state. LangChain lives in
the optional `offline` uv group and is never installed in the runtime image;
this script is its only use.

Transaction shape: the LangChain store commits per call on its own SQLAlchemy
connection, so its engine is fed a shared psycopg AsyncConnection whose
commit/rollback/close are neutralised — upserts, deletes and the index_state
write land in one explicit transaction. The whole push runs under the same
advisory-lock key as PgIndexStore.sync (session-scoped here so it also covers
the schema ensure and post-commit maintenance outside the data transaction).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.contract import CONTRACT_VERSION
from app.ingest import embed_documents, load_documents, source_generated_at
from app.pg_common import SYNC_LOCK_KEY, diff_documents
from app.pg_index import PgIndexStore

logger = logging.getLogger("ceit.sidecar.push")

# Columns PgIndexStore owns outside the document payload; AsyncPGVectorStore
# fills metadata_columns from doc.metadata and drops those keys from the
# metadata jsonb, so the stored rows match a sync exactly.
_METADATA_COLUMNS = ("corpus", "title", "content_hash")


class _HeldConnection(psycopg.AsyncConnection):
    """psycopg AsyncConnection that SQLAlchemy cannot commit, roll back or
    close — LangChain's per-call commits become no-ops and every statement
    stays inside the explicit transaction the script owns."""

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass


# --- async mirrors of the PgIndexStore state reads/write (same SQL; the sync
# versions cannot run on this connection) -------------------------------------


async def _stored_hashes(conn, docs: sql.Identifier) -> dict:
    cur = await conn.execute(sql.SQL("SELECT id, content_hash FROM {}").format(docs))
    return dict(await cur.fetchall())


async def _state_row(conn, state: sql.Identifier) -> dict | None:
    cur = await conn.execute(
        sql.SQL(
            "SELECT version, built_at, source_generated_at, model_name,"
            " contract_version, keyword_index FROM {} WHERE id = 1"
        ).format(state)
    )
    row = await cur.fetchone()
    if row is None:
        return None
    return {
        "version": row[0],
        "built_at": row[1],
        "source_generated_at": row[2],
        "model_name": row[3],
        "contract_version": row[4],
        "keyword_index": row[5],
    }


async def _live_counts(conn, docs: sql.Identifier) -> dict:
    """Truth from the documents table — same query as PgIndexStore._live_counts."""
    cur = await conn.execute(
        sql.SQL("SELECT corpus, count(*), count(embedding) FROM {} GROUP BY corpus").format(docs)
    )
    rows = await cur.fetchall()
    by_corpus = {corpus: total for corpus, total, _embedded in rows}
    return {
        "documents": sum(by_corpus.values()),
        "embedded": sum(embedded for _corpus, _total, embedded in rows),
        "by_corpus": by_corpus,
    }


async def _write_index_state(
    conn,
    state_ident: sql.Identifier,
    state_row: dict | None,
    dirty: bool,
    source_generated: str | None,
    counts: dict,
    model_name: str,
    keyword_index: str,
) -> dict:
    # version/built_at advance only when the data changed; metadata is
    # refreshed either way (same rule as PgIndexStore._write_state).
    if dirty or state_row is None:
        version = (state_row["version"] if state_row else 0) + 1
        built_at = datetime.now(UTC)
    else:
        version, built_at = state_row["version"], state_row["built_at"]
    state = {
        "version": version,
        "built_at": built_at,
        "source_generated_at": source_generated,
        "documents": counts["documents"],
        "embedded": counts["embedded"],
        "by_corpus": Jsonb(counts["by_corpus"]),
        "model_name": model_name,
        "contract_version": CONTRACT_VERSION,
        "keyword_index": keyword_index,
    }
    await conn.execute(
        sql.SQL(
            """INSERT INTO {}
                (id, version, built_at, source_generated_at, documents,
                 embedded, by_corpus, model_name, contract_version,
                 keyword_index)
                VALUES (1, %(version)s, %(built_at)s, %(source_generated_at)s,
                    %(documents)s, %(embedded)s, %(by_corpus)s, %(model_name)s,
                    %(contract_version)s, %(keyword_index)s)
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
        ).format(state_ident),
        state,
    )
    state["by_corpus"] = counts["by_corpus"]
    return state


async def _push_async(
    conn: _HeldConnection,
    docs: list[dict],
    *,
    schema: str,
    model_name: str,
    force_full: bool,
    embed_fn,
    source_generated: str | None,
    keyword_index: str,
) -> dict:
    # The LangChain stack is imported lazily so `import` (and the runtime
    # image, which never installs it) stays clean of the offline group.
    from langchain_core.embeddings import Embeddings
    from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
    from langchain_postgres.v2.engine import PGEngine
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    class _UnusedEmbeddings(Embeddings):
        """PGVectorStore requires an embedding service; it is never invoked —
        vectors come from the app's own embedder."""

        def embed_documents(self, texts):
            raise NotImplementedError

        def embed_query(self, text):
            raise NotImplementedError

    async def _creator():
        # Every engine connection is the held connection, so LangChain's
        # upserts/deletes execute inside the script's transaction.
        return conn

    # The URL only selects the psycopg async dialect — _creator supplies the
    # real connection, so the host/database here are placeholders.
    sa_engine = create_async_engine(
        "postgresql+psycopg://127.0.0.1/push",
        async_creator=_creator,
        poolclass=NullPool,
    )
    docs_ident = sql.Identifier(schema, "documents")
    state_ident = sql.Identifier(schema, "index_state")

    await conn.execute("BEGIN")
    try:
        stored = await _stored_hashes(conn, docs_ident)
        added, changed, removed, wanted = diff_documents(stored, docs, model_name)
        to_embed = list(docs) if force_full else added + changed
        dirty = bool(to_embed or removed)

        store = None
        if to_embed or removed:
            store = await AsyncPGVectorStore.create(
                PGEngine.from_engine(sa_engine),
                _UnusedEmbeddings(),
                "documents",
                schema_name=schema,
                id_column="id",
                content_column="text",
                embedding_column="embedding",
                metadata_json_column="metadata",
                metadata_columns=list(_METADATA_COLUMNS),
            )
        if to_embed:
            vectors = (
                embed_fn(to_embed)
                if embed_fn is not None
                else embed_documents(to_embed, model_name)
            )
            if len(vectors) != len(to_embed):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(to_embed)} documents"
                )
            await store.aadd_embeddings(
                texts=[d["text"] for d in to_embed],
                embeddings=[[float(x) for x in v] for v in vectors],
                metadatas=[
                    {
                        **d["metadata"],
                        "corpus": d["corpus"],
                        "title": d["title"],
                        "content_hash": wanted[d["id"]],
                    }
                    for d in to_embed
                ],
                ids=[d["id"] for d in to_embed],
            )
        if removed:
            await store.adelete(ids=removed)

        state_row = await _state_row(conn, state_ident)
        counts = await _live_counts(conn, docs_ident)
        state = await _write_index_state(
            conn,
            state_ident,
            state_row,
            dirty,
            source_generated,
            counts,
            model_name,
            keyword_index,
        )
        await conn.execute("COMMIT")
    except BaseException:
        await conn.execute("ROLLBACK")
        raise
    finally:
        await sa_engine.dispose()
    return {
        **state,
        "built_at": state["built_at"].isoformat(),
        "added": len(added),
        "changed": len(changed),
        "removed": len(removed),
        "embedded_now": len(to_embed),
        "dirty": dirty,
    }


def push(
    corpus_path: Path,
    *,
    database_url: str,
    schema: str = "public",
    model_name: str = "all-MiniLM-L6-v2",
    force_full: bool = False,
    embed_fn=None,
    connect_timeout: int = 15,
) -> dict:
    """Push the corpus into <schema>.documents; returns the new index state.

    Mirrors PgIndexStore.sync: validate before connecting, content-hash diff
    (--force-full re-embeds everything), embed only added/changed rows, one
    transaction for every write, then post-commit keyword-index + VACUUM
    maintenance. An unchanged corpus is a complete no-op.
    """
    corpus_path = Path(corpus_path)
    docs = load_documents(corpus_path)
    source_generated = source_generated_at(corpus_path)

    store = PgIndexStore(database_url, schema=schema, connect_timeout=connect_timeout)
    # psycopg async needs a selector loop; Windows' default proactor loop is
    # rejected at connect time.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def _run() -> dict:
        conn = await _HeldConnection.connect(
            database_url,
            autocommit=True,
            prepare_threshold=None,
            connect_timeout=connect_timeout,
        )
        try:
            return await _push_async(
                conn,
                docs,
                schema=schema,
                model_name=model_name,
                force_full=force_full,
                embed_fn=embed_fn,
                source_generated=source_generated,
                keyword_index=keyword_index,
            )
        finally:
            # _HeldConnection.close() is neutralised — close for real.
            await psycopg.AsyncConnection.close(conn)

    with store._connect(autocommit=True) as conn:
        # Session-scoped lock (same key space as sync's advisory xact lock) —
        # PgIndexStore's private helpers are shared deliberately so schema,
        # diff and maintenance can never drift between the two writers.
        # the schema ensure, data transaction and maintenance all serialise
        # against a concurrent PgIndexStore.sync or second push.
        conn.execute("SELECT pg_advisory_lock(%s)", (SYNC_LOCK_KEY,))
        try:
            store._ensure_schema(conn)
            lakebase = store._lakebase_available(conn)
            keyword_index = "lakebase_bm25" if lakebase else "gin"

            state = asyncio.run(_run())

            # Same post-commit rule as sync: maintenance failures must not
            # fail an already-committed push.
            try:
                store._post_commit_maintenance(lakebase, state["dirty"])
            except psycopg.Error:
                logger.warning("push: post-commit maintenance failed", exc_info=True)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (SYNC_LOCK_KEY,))

    logger.info(
        "push: added=%d changed=%d removed=%d embedded=%d version=%d",
        state["added"],
        state["changed"],
        state["removed"],
        state["embedded_now"],
        state["version"],
    )
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed the corpus with FastEmbed and push it into Postgres"
        " via the LangChain vector store (offline `uv` group required)."
    )
    parser.add_argument(
        "corpus",
        type=Path,
        help="corpus export directory holding catalog.json/policies.json"
        " (a file inside it also works)",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="re-embed and rewrite every document, ignoring the hash diff",
    )
    args = parser.parse_args(argv)

    from app.config import settings

    if not settings.database_url:
        parser.error("DATABASE_URL is not set")
    corpus_path = args.corpus if args.corpus.is_dir() else args.corpus.parent
    try:
        state = push(
            corpus_path,
            database_url=settings.database_url,
            schema=settings.pg_index_schema,
            model_name=settings.model_name,
            force_full=args.force_full,
            connect_timeout=settings.pg_connect_timeout,
        )
    except ImportError:
        parser.exit(
            1, "error: langchain-postgres is not installed — run via `uv run --group offline`\n"
        )
    print(
        f"pushed: version={state['version']} documents={state['documents']}"
        f" embedded={state['embedded_now']} (added={state['added']}"
        f" changed={state['changed']} removed={state['removed']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
