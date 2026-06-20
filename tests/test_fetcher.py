"""
tests/test_fetcher.py
=====================
Tests for data/fetcher.fetch_and_cache — the cache-first memoiser behind the
real-data notebooks.  Network-free: the fetch is a stub closure, so this runs in
CI.  Anchors: a cache miss runs the closure and writes the dated parquet; a hit
reads the cache and does NOT run the closure again (offline reproducibility).
"""

from __future__ import annotations

import pandas as pd

from data.fetcher import fetch_and_cache


def test_miss_writes_then_hit_reads(tmp_path) -> None:
    calls = {"n": 0}
    df0 = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "iv": [0.22, 0.20, 0.19]})

    def fetch_fn():
        calls["n"] += 1
        return df0

    # miss: runs fetch_fn once and writes the dated parquet
    out1 = fetch_and_cache("SPY", "demo", "2026-06-20", fetch_fn, cache_dir=tmp_path)
    assert calls["n"] == 1
    assert (tmp_path / "SPY_demo_2026-06-20.parquet").exists()
    pd.testing.assert_frame_equal(out1, df0)

    # hit: reads cache, fetch_fn is NOT called again, data is identical
    out2 = fetch_and_cache("SPY", "demo", "2026-06-20", fetch_fn, cache_dir=tmp_path)
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(out2, df0)
