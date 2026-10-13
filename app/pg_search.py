"""PgVectorSearch: SearchBackend over the Postgres index (#7).

Same semantics as the SQLite backend: BM25 over all matches with OR of
tokens, the relevance gate over the whole UNFILTERED index, filters applied
per channel before ranking (in SQL here, in Python for sqlite), shared RRF
fusion and the exact-code pin.

The keyword channel uses the lakebase_bm25 <@> index when index_state says
it exists, falling back to ts_rank_cd + @@ match on plain pgvector — the
fallback is functional but not score-identical (measured in the #5 spike).
Per-query settings are SET LOCAL inside the query transaction, safe under
Neon's transaction pooling; connections come from a psycopg_pool so search
never pays a connect handshake. recommend() lands in #11.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool
from sqlitesearch import Tokenizer

from .config import settings
from .fusion import RRF_K, fuse, pin_catalog_code, shape_results
from .pg_index import _vector_literal

# Same query-side tokenizer as the sqlite backend: split on non-word,
# keep digits (years, catalog codes), drop English stop words.
_TOKENIZER = Tokenizer(pattern=r"[\s\W]+", stop_words="english")
# lakebase_bm25.default_limit accepts at most 65535 (spike finding).
_BM25_MAX_ROWS = 65535


def _query_tokens(query: str) -> list[str]:
    """Tokenize like the sqlite backend; [] when no token can match."""
    if not re.search(r"[A-Za-z0-9]", query):
        return []
    return [t for t in _TOKENIZER.tokenize(query) if t]


def _like_escape(needle: str) -> str:
    """Escape ILIKE metachars so `%%needle%%` is a literal substring match."""
    return needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filter_sql(filters: dict | None, corpus: str | None) -> tuple[list, list]:
    """passes_filters() as SQL predicates — same keys, applied before ranking.

    A non-numeric publication_year never matches (sqlite treats it as != or
    0), so casts are guarded by a digits regex instead of erroring.
    """
    where: list[sql.Composable] = []
    params: list = []
    f = filters or {}
    if f.get("paper_type"):
        where.append(sql.SQL("metadata->>'paper_type' = %s"))
        params.append(f["paper_type"])
    if f.get("department"):
        where.append(sql.SQL("metadata->>'department' = %s"))
        params.append(f["department"])
    digits = sql.SQL("(metadata->>'publication_year') ~ '^[0-9]+$'")
    if f.get("publication_year"):
        where.append(sql.SQL("{} AND (metadata->>'publication_year')::int = %s").format(digits))
        params.append(int(f["publication_year"]))
    if f.get("year_from"):
        where.append(sql.SQL("{} AND (metadata->>'publication_year')::int >= %s").format(digits))
        params.append(int(f["year_from"]))
    if f.get("year_to"):
        where.append(sql.SQL("{} AND (metadata->>'publication_year')::int <= %s").format(digits))
        params.append(int(f["year_to"]))
    if f.get("author"):
        where.append(
            sql.SQL(
                "EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'authors') a"
                " WHERE a ILIKE %s ESCAPE '\\')"
            )
        )
        params.append(f"%{_like_escape(str(f['author']))}%")
    if f.get("adviser"):
        where.append(
            sql.SQL(
                "(metadata->>'research_adviser' ILIKE %s ESCAPE '\\'"
                " OR metadata->>'technical_adviser' ILIKE %s ESCAPE '\\')"
            )
        )
        params.extend([f"%{_like_escape(str(f['adviser']))}%"] * 2)
    if corpus:
        where.append(sql.SQL("corpus = %s"))
        params.append(corpus)
    return where, params


def _where(where: list, extra: sql.SQL | None = None) -> sql.Composed:
    """Compose `WHERE p1 AND p2 ...`; `extra` is the leading predicate when
    the caller adds its own (e.g. the @@ match on the GIN fallback)."""
    parts = ([extra] if extra is not None else []) + where
    if not parts:
        return sql.SQL("")
    return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(parts)


class PgVectorSearch:
    """SearchBackend over the Postgres documents table."""

    def __init__(
        self,
        database_url: str,
        schema: str = "public",
        model_name: str = "all-MiniLM-L6-v2",
        pool_size: int = 4,
        connect_timeout: int = 15,
        embed_query_fn: Callable[[str], np.ndarray] | None = None,
    ):
        self.schema = schema
        self.model_name = model_name
        self._docs = sql.Identifier(schema, "documents")
        self._state = sql.Identifier(schema, "index_state")
        self._bm25_index = sql.Identifier(schema, "documents_bm25").as_string()
        self._embed_query = embed_query_fn
        # Lazy open: constructing the engine must not need the database —
        # a sleeping Neon compute or a dead URL only fails on first use.
        self._pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=pool_size,
            kwargs={"prepare_threshold": None, "connect_timeout": connect_timeout},
            open=False,
        )
        self._pool_open = False

    def close(self) -> None:
        if self._pool_open:
            self._pool.close()
            self._pool_open = False

    def _conn(self):
        if not self._pool_open:
            self._pool.open(wait=False)
            self._pool_open = True
        return self._pool.connection()

    def _embed(self, query: str) -> np.ndarray:
        if self._embed_query is not None:
            return self._embed_query(query)
        from .ingest import embed_query

        return embed_query(query, self.model_name)

    def _index_status(self, conn) -> tuple[str | None, bool]:
        """(keyword_index, has_docs) from one snapshot — None state row means
        no sync has ever committed, so search returns empty like sqlite."""
        row = conn.execute(
            sql.SQL(
                "SELECT (SELECT keyword_index FROM {} WHERE id = 1), EXISTS(SELECT 1 FROM {})"
            ).format(self._state, self._docs)
        ).fetchone()
        return row[0], bool(row[1])

    def _bm25_ranks(
        self, conn, tokens: list[str], where: list, params: list, lakebase: bool
    ) -> dict[str, int]:
        text = " ".join(tokens)
        if lakebase:
            try:
                # SET LOCAL scopes the GUCs to this transaction only — safe
                # through Neon's transaction-mode pooler (spike finding).
                with conn.transaction():
                    conn.execute(f"SET LOCAL lakebase_bm25.default_limit = {_BM25_MAX_ROWS}")
                    conn.execute("SET LOCAL lakebase_bm25.prefilter = on")
                    rows = conn.execute(
                        sql.SQL(
                            "SELECT id, score FROM ("
                            " SELECT id, tsv <@> to_bm25query("
                            "  to_tsvector('simple', %s), %s::regclass) AS score"
                            " FROM {}{} ORDER BY score"
                            ") t WHERE score < 0 ORDER BY score, id"
                        ).format(self._docs, _where(where)),
                        (text, self._bm25_index, *params),
                    ).fetchall()
                return {doc_id: rank for rank, (doc_id, _s) in enumerate(rows, start=1)}
            except psycopg.errors.UndefinedObject:
                # lakebase_text or the index vanished after state was written
                # — degrade to the GIN path rather than 500.
                pass
        or_query = " | ".join(tokens)
        rows = conn.execute(
            sql.SQL(
                "SELECT id, -ts_rank_cd(tsv, to_tsquery('simple', %s)) AS score"
                " FROM {}{} ORDER BY score, id"
            ).format(
                self._docs,
                _where(where, extra=sql.SQL("tsv @@ to_tsquery('simple', %s)")),
            ),
            (or_query, or_query, *params),
        ).fetchall()
        return {doc_id: rank for rank, (doc_id, _s) in enumerate(rows, start=1)}

    def _semantic_scores(self, conn, query: str, where: list, params: list):
        vec = _vector_literal(self._embed(query))
        # The gate is over the WHOLE index — an off-corpus query must not
        # surface nearest neighbours just because filters narrowed the pool.
        max_cos = conn.execute(
            sql.SQL("SELECT max(1 - (embedding <=> %s::vector)) FROM {}").format(self._docs),
            (vec,),
        ).fetchone()[0]
        if max_cos is None or max_cos < settings.min_semantic_similarity:
            return {}
        rows = conn.execute(
            sql.SQL("SELECT id, 1 - (embedding <=> %s::vector) FROM {}{}").format(
                self._docs, _where(where)
            ),
            (vec, *params),
        ).fetchall()
        return {doc_id: float(score) for doc_id, score in rows}

    def rrf_search(
        self,
        query: str,
        k: int = RRF_K,
        limit: int = 10,
        filters: dict | None = None,
        corpus: str | None = None,
        include_text: bool = False,
        method: str = "hybrid",
    ) -> list[dict]:
        method = method if method in ("bm25", "semantic") else "hybrid"
        with self._conn() as conn:
            try:
                keyword_index, has_docs = self._index_status(conn)
            except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName):
                conn.rollback()
                return []
            if not has_docs:
                return []

            where, params = _filter_sql(filters, corpus)
            tokens = _query_tokens(query)
            bm25_ranks = (
                self._bm25_ranks(conn, tokens, where, params, keyword_index == "lakebase_bm25")
                if method in ("bm25", "hybrid") and tokens
                else {}
            )
            semantic_scores = (
                self._semantic_scores(conn, query, where, params)
                if method in ("semantic", "hybrid")
                else {}
            )

            ranked, scores, sem_rank_map = fuse(method, k, bm25_ranks, semantic_scores)
            if not ranked:
                return []

            docs_by_id = {
                r[0]: {"id": r[0], "corpus": r[1], "title": r[2], "metadata": r[3] or {}}
                for r in conn.execute(
                    sql.SQL("SELECT id, corpus, title, metadata FROM {} WHERE id = ANY(%s)").format(
                        self._docs
                    ),
                    (ranked,),
                ).fetchall()
            }
            pinned_id = pin_catalog_code(query, ranked, docs_by_id) if method == "hybrid" else None

            if include_text:
                top_ids = ranked[:limit]
                for doc_id, text in conn.execute(
                    sql.SQL("SELECT id, text FROM {} WHERE id = ANY(%s)").format(self._docs),
                    (top_ids,),
                ).fetchall():
                    docs_by_id[doc_id]["text"] = text

            return shape_results(
                ranked,
                limit,
                docs_by_id,
                scores,
                bm25_ranks,
                sem_rank_map,
                method,
                pinned_id,
                include_text,
            )

    def recommend(self, seed_ids: list[str], exclude_ids: list[str], limit: int):
        raise NotImplementedError("pgvector recommend lands in #11")
