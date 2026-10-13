"""THROWAWAY SPIKE (sidecar #5): lakebase_text BM25 parity on Neon. Not production code.

Loads the committed corpus into a Neon *branch*, builds a lakebase_bm25 index on a
tsvector ('simple' config) and measures the questions in issue #5. Two tsvector
variants are compared: "raw" (to_tsvector('simple', body)) and "normalized"
(non-alphanumerics replaced by spaces first, mimicking SQLite FTS5 unicode61).

The connection string is read from SPIKE_DATABASE_URL and is never printed or stored.

    SPIKE_DATABASE_URL=... python scripts/spike_neon_bm25.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SIDECAR_TOKEN", "spike")

from sqlitesearch import Tokenizer

from app.config import settings
from app.eval import aggregate, category_of, evaluate_case, load_golden
from app.ingest import load_documents
from app.search import HybridSearch

TOKENIZER = Tokenizer(pattern=r"[\s\W]+", stop_words="english")
MAX_DEFAULT_LIMIT = 65535
VARIANTS = {
    "raw": "to_tsvector('simple', body)",
    "normalized": "to_tsvector('simple', regexp_replace(body, '[^[:alnum:]]+', ' ', 'g'))",
}
SCORE = "tsv <@> to_bm25query(to_tsvector('simple', %s), 'documents_bm25'::regclass)"


def connect() -> psycopg.Connection:
    return psycopg.connect(
        os.environ["SPIKE_DATABASE_URL"], autocommit=True, prepare_threshold=None
    )


def load(conn: psycopg.Connection, tsv_expr: str) -> int:
    docs = load_documents(ROOT / "corpus")
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        "CREATE TABLE documents (id text PRIMARY KEY, corpus text NOT NULL, title text NOT NULL,"
        " body text NOT NULL, department text, paper_type text, catalog_code text,"
        f" tsv tsvector GENERATED ALWAYS AS ({tsv_expr}) STORED)"
    )
    rows = []
    for d in docs:
        m = d.get("metadata") or {}
        rows.append(
            (
                d["id"],
                d["corpus"],
                d["title"],
                d["text"],
                m.get("department"),
                m.get("paper_type"),
                m.get("catalog_code"),
            )
        )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (id, corpus, title, body, department, paper_type, catalog_code)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            rows,
        )
    conn.execute(
        "CREATE INDEX documents_bm25 ON documents USING lakebase_bm25 (tsv)"
        " WITH (k1 = 1.2, b = 0.75)"
    )
    conn.execute("VACUUM documents")
    return len(docs)


def query_text(query: str) -> str | None:
    if not re.search(r"[A-Za-z0-9]", query):
        return None
    return " ".join(t for t in TOKENIZER.tokenize(query) if t) or None


def or_tsquery(text: str) -> str:
    return " | ".join(text.split())


class PgEngine:
    """SearchBackend-shaped engine so app.eval.evaluate_case scores Postgres like SQLite.

    mode "index": the production-shaped path - ORDER BY <@> LIMIT through the BM25 index,
    inner limit 500, zero-score rows dropped, deterministic (score, id) final order.
    mode "fallback": built-in ts_rank_cd with a WHERE @@ match, no extension.
    """

    def __init__(self, conn, mode="index", prefilter=False):
        self.conn, self.mode, self.prefilter = conn, mode, prefilter

    def rrf_search(
        self, query, k=60, limit=10, filters=None, corpus=None, include_text=False, method="bm25"
    ):
        text = query_text(query)
        if text is None:
            return []
        where, params = [], []
        if corpus:
            where.append("corpus = %s")
            params.append(corpus)
        for key, value in (filters or {}).items():
            where.append(f"{key} = %s")
            params.append(value)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self.conn.transaction(), self.conn.cursor() as cur:
            if self.mode == "index":
                cur.execute(f"SET LOCAL lakebase_bm25.default_limit = {MAX_DEFAULT_LIMIT}")
                if self.prefilter:
                    cur.execute("SET LOCAL lakebase_bm25.prefilter = on")
                cur.execute(
                    f"SELECT id, score FROM (SELECT id, {SCORE} AS score FROM documents {clause}"
                    " ORDER BY score LIMIT 500) t WHERE score < 0 ORDER BY score, id LIMIT %s",
                    [text, *params, limit],
                )
            else:
                and_ = (" AND " + " AND ".join(where)) if where else ""
                cur.execute(
                    "SELECT id, -ts_rank_cd(tsv, to_tsquery('simple', %s)) AS score FROM documents"
                    f" WHERE tsv @@ to_tsquery('simple', %s){and_} ORDER BY score, id LIMIT %s",
                    [or_tsquery(text), or_tsquery(text), *params, limit],
                )
            return [{"id": r[0], "score": r[1]} for r in cur.fetchall()]

    def close(self) -> None:
        pass


def golden(engine, limit: int = 5):
    cases = load_golden()["test_cases"]
    results = [evaluate_case(engine, c, limit, "bm25") for c in cases]
    return aggregate(results), results, cases


def parity_gap(pg_results, sqlite_results, cases) -> dict:
    """Per-category and per-case differences versus the live SQLite FTS5 BM25 run."""
    by_cat: dict[str, dict] = {}
    worse = []
    for case, pg, sq in zip(cases, pg_results, sqlite_results):
        if pg["negative"]:
            continue
        cat = category_of(case)
        d = by_cat.setdefault(
            cat, {"n": 0, "recall_delta": 0.0, "precision_delta": 0.0, "top1_lost": 0}
        )
        d["n"] += 1
        d["recall_delta"] += pg["recall"] - sq["recall"]
        d["precision_delta"] += pg["precision"] - sq["precision"]
        d["top1_lost"] += int(sq["top1_hit"] and not pg["top1_hit"])
        if pg["recall"] < sq["recall"] or (sq["top1_hit"] and not pg["top1_hit"]):
            worse.append(
                {
                    "query": case["query"][:60],
                    "category": cat,
                    "sqlite_recall": sq["recall"],
                    "pg_recall": pg["recall"],
                    "sqlite_top1": sq["top1_hit"],
                    "pg_top1": pg["top1_hit"],
                }
            )
    for d in by_cat.values():
        d["recall_delta"] = round(d["recall_delta"] / d["n"], 4)
        d["precision_delta"] = round(d["precision_delta"] / d["n"], 4)
    return {"by_category_avg_delta(pg-sqlite)": by_cat, "cases_where_pg_is_worse": worse}


def environment(conn) -> dict:
    out = {"server_version": conn.execute("SHOW server_version").fetchone()[0]}
    out["available_lakebase_extensions"] = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM pg_available_extensions WHERE name LIKE 'lakebase%' ORDER BY 1"
        )
    ]
    conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_text")
    out["lakebase_text"] = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname='lakebase_text'"
    ).fetchone()[0]
    out["default_limit_default"] = conn.execute("SHOW lakebase_bm25.default_limit").fetchone()[0]
    out["prefilter_default"] = conn.execute("SHOW lakebase_bm25.prefilter").fetchone()[0]
    for bad in (65536, 1_000_000):
        try:
            with conn.transaction():
                conn.execute(f"SET LOCAL lakebase_bm25.default_limit = {bad}")
            out[f"default_limit={bad}"] = "accepted"
        except psycopg.errors.InvalidParameterValue as exc:
            out[f"default_limit={bad}"] = str(exc).splitlines()[0].split(" for ")[0][:80]
    return out


def tokenization(conn) -> dict:
    out = {}
    for code in ("CEIT-EE-03-07", "CEIT-CE-18-01"):
        out[f"query-side to_tsvector('{code}')"] = conn.execute(
            "SELECT to_tsvector('simple', %s)::text", [code]
        ).fetchone()[0]
    out["sqlite FTS5 / sidecar tokenizer on CEIT-EE-03-07"] = TOKENIZER.tokenize("CEIT-EE-03-07")
    row = conn.execute(
        "SELECT tsv::text FROM documents WHERE catalog_code='CEIT-CE-18-01'"
    ).fetchone()
    out["stored doc lexemes containing ceit / 01 / 18"] = [
        t for t in row[0].split(" ") if any(x in t for x in ("ceit", "'01'", "-01", "18"))
    ]
    out["can code query 'CEIT-EE-03-07' retrieve its own doc (top-1)?"] = bool(
        conn.execute(
            f"SELECT id FROM documents WHERE catalog_code='CEIT-EE-03-07' AND id = (SELECT id FROM"
            f" documents ORDER BY {SCORE}, id LIMIT 1)",
            [query_text("CEIT-EE-03-07")],
        ).fetchone()
    )
    return out


def plan_node(conn, sql, params) -> str:
    lines = [r[0] for r in conn.execute("EXPLAIN " + sql, params)]
    return next((l.strip().split("  (")[0] for l in lines if "Scan" in l), lines[0])


def force_index(conn) -> None:
    conn.execute("SET LOCAL enable_seqscan = off")


def seqscan_scores(conn, where: str, text: str) -> list[float]:
    """Ground truth: exact BM25 scores for rows passing `where`, via a plain seq scan."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        rows = conn.execute(
            f"SELECT {SCORE} AS s FROM documents WHERE {where} AND tsv @@ to_tsquery('simple', %s)",
            [text, or_tsquery(text)],
        ).fetchall()
    return sorted(r[0] for r in rows)


def candidate_limit(conn) -> dict:
    """Does a high default_limit return every matching row through the INDEX path?"""
    text = "paper section and"
    truth = seqscan_scores(conn, "true", text)
    sql = f"SELECT id, {SCORE} AS s FROM documents ORDER BY s LIMIT 100000"
    with conn.transaction():
        force_index(conn)
        plan = plan_node(conn, sql, [text])
    out = {
        "query_terms": text.split(),
        "rows_matching_exact(@@)": len(truth),
        "total_rows": conn.execute("SELECT count(*) FROM documents").fetchone()[0],
        "plan_with_seqscan_off": plan,
    }
    for label, value in (("default(1000)", None), ("65535", 65535), ("-1", -1)):
        try:
            with conn.transaction():
                force_index(conn)
                if value is not None:
                    conn.execute(f"SET LOCAL lakebase_bm25.default_limit = {value}")
                rows = conn.execute(sql, [text]).fetchall()
        except psycopg.Error as exc:
            out[f"default_limit={label}"] = f"ERROR: {str(exc).splitlines()[0][:120]}"
            continue
        scored = [r[1] for r in rows if r[1] < 0]
        out[f"default_limit={label}"] = {
            "rows_returned": len(rows),
            "rows_with_score<0": len(scored),
            "returns_every_matching_row": len(scored) == len(truth),
            "scores_equal_seqscan_truth": sorted(scored) == truth,
        }
    return out


def prefilter_check(conn) -> dict:
    """Corpus/metadata filters through the INDEX path, prefilter off vs on, tie-safe by score."""
    text = query_text("engineering research")
    where = "corpus = 'catalog' AND department = 'Civil Engineering' AND paper_type = 'Research'"
    truth = seqscan_scores(conn, where, text)[:20]
    sql = f"SELECT id, {SCORE} AS s FROM documents WHERE {where} ORDER BY s LIMIT 20"
    with conn.transaction():
        force_index(conn)
        plan = plan_node(conn, sql, [text])
    out = {
        "filters": where,
        "matching_rows_in_filter": len(seqscan_scores(conn, where, text)),
        "truth_top20_scores_from_seqscan": len(truth),
        "plan_with_seqscan_off": plan,
    }
    for prefilter in ("off", "on"):
        for dl in (20, 200, MAX_DEFAULT_LIMIT):
            with conn.transaction():
                force_index(conn)
                conn.execute(f"SET LOCAL lakebase_bm25.default_limit = {dl}")
                conn.execute(f"SET LOCAL lakebase_bm25.prefilter = {prefilter}")
                got = [(r[0], r[1]) for r in conn.execute(sql, [text]) if r[1] < 0]
            ids = [g[0] for g in got]
            out[f"prefilter={prefilter} default_limit={dl}"] = {
                "returned": len(got),
                "all_pass_filter": bool(ids)
                and bool(
                    conn.execute(
                        "SELECT bool_and(" + where + ") FROM documents WHERE id = ANY(%s)", [ids]
                    ).fetchone()[0]
                ),
                "top20_scores_equal_truth": sorted(g[1] for g in got) == truth,
            }
    return out


def ordering_and_zero(conn) -> dict:
    text = query_text("energy efficient street lighting")
    sql = f"SELECT id, {SCORE} AS s FROM documents ORDER BY s LIMIT 100000"
    with conn.transaction():
        force_index(conn)
        conn.execute(f"SET LOCAL lakebase_bm25.default_limit = {MAX_DEFAULT_LIMIT}")
        plan_used = plan_node(conn, sql, [text])
        rows = conn.execute(sql, [text]).fetchall()
        rows2 = conn.execute(sql, [text]).fetchall()
    scores = [r[1] for r in rows]
    matching = conn.execute(
        "SELECT count(*) FROM documents WHERE tsv @@ to_tsquery('simple', %s)", [or_tsquery(text)]
    ).fetchone()[0]
    ties = {}
    for _id, s in rows:
        if s < 0:
            ties.setdefault(s, []).append(_id)
    return {
        "plan_with_seqscan_off": plan_used,
        "rows_returned": len(rows),
        "matching_rows(@@)": matching,
        "rows_with_score<0": sum(1 for s in scores if s < 0),
        "rows_with_score==0": sum(1 for s in scores if s == 0),
        "rows_with_score>0": sum(1 for s in scores if s > 0),
        "ascending_(most_negative_first)": scores == sorted(scores),
        "best_3": [(r[0], round(r[1], 3)) for r in rows[:3]],
        "tied_score_groups_among_matches": sum(1 for v in ties.values() if len(v) > 1),
        "identical_order_on_repeat": [r[0] for r in rows] == [r[0] for r in rows2],
        "zero_score_rows_are_returned_so_a_score<0_filter_is_required": any(s == 0 for s in scores),
    }


def run_variant(conn, name: str, sqlite_results, sqlite_cases, sqlite_top5, cases_all) -> dict:
    report: dict = {"rows_loaded": load(conn, VARIANTS[name])}
    report["tokenization"] = tokenization(conn)
    report["candidate_limit"] = candidate_limit(conn)
    report["prefilter"] = prefilter_check(conn)
    report["ordering_and_zero_score"] = ordering_and_zero(conn)
    agg, res, _ = golden(PgEngine(conn, "index"))
    report["golden_bm25_index_k5"] = agg
    report["top1_differences_vs_sqlite"] = [
        {
            "query": c["query"][:70],
            "relevant": c.get("relevant_docs"),
            "sqlite_top5": sqlite_top5[c["query"]],
            "pg_top5": [
                x["id"]
                for x in PgEngine(conn, "index").rrf_search(
                    c["query"], limit=5, corpus=c.get("corpus")
                )
            ],
        }
        for c, pg, sq in zip(cases_all, res, sqlite_results)
        if not pg["negative"] and pg["top1_hit"] != sq["top1_hit"]
    ]
    report["parity_gap_vs_sqlite"] = parity_gap(res, sqlite_results, sqlite_cases)
    report["golden_bm25_index_k5_prefilter_on"] = golden(PgEngine(conn, "index", prefilter=True))[0]
    report["golden_fallback_ts_rank_cd_k5"] = golden(PgEngine(conn, "fallback"))[0]
    return report


def main() -> int:
    sqlite = HybridSearch(Path(settings.cache_dir), settings.model_name)
    cases = load_golden()["test_cases"]
    sqlite_results = [evaluate_case(sqlite, c, 5, "bm25") for c in cases]
    sqlite_top5 = {
        c["query"]: [
            r["id"]
            for r in sqlite.rrf_search(c["query"], limit=5, corpus=c.get("corpus"), method="bm25")
        ]
        for c in cases
    }
    conn = connect()
    report = {
        "sqlite_fts5_bm25_k5_live": aggregate(sqlite_results),
        "environment": environment(conn),
    }
    for name in VARIANTS:
        report[f"variant_{name}"] = run_variant(
            conn, name, sqlite_results, cases, sqlite_top5, cases
        )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
