"""Warm latency + memory benchmark replaying the golden queries (baseline, SQLite backend)."""

import ctypes
import json
import statistics
import time
from ctypes import wintypes
from pathlib import Path

from app.config import settings
from app.eval import RRF_K, load_golden
from app.rerank import Reranker
from app.search import HybridSearch


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem_mb():
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PMC),
        wintypes.DWORD,
    ]
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb
    )
    return round(p.WorkingSetSize / 2**20, 1), round(p.PeakWorkingSetSize / 2**20, 1)


PASSES = 5
cases = load_golden()["test_cases"]
rss_before = mem_mb()
engine = HybridSearch(Path(settings.cache_dir), settings.model_name)
engine.rrf_search("warmup", k=RRF_K, limit=5, method="hybrid")
rss_warm = mem_mb()
rr = Reranker(mode="blend")
out = {
    "passes": PASSES,
    "queries": len(cases),
    "rss_mb_before_load": rss_before[0],
    "rss_mb_after_warmup": rss_warm,
}
for name, method, rerank in [
    ("bm25", "bm25", False),
    ("semantic", "semantic", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]:
    times = []
    for _ in range(PASSES):
        for c in cases:
            t = time.perf_counter()
            r = engine.rrf_search(
                c["query"],
                k=RRF_K,
                limit=5,
                filters=c.get("filters") or {},
                corpus=c.get("corpus"),
                method=method,
            )
            if rerank and r:
                rr.rerank(c["query"], r)
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    out[name] = {
        "p50_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 1),
        "n": len(times),
    }
out["rss_mb_end"], out["peak_rss_mb"] = mem_mb()[0], mem_mb()[1]
print(json.dumps(out, indent=2))
