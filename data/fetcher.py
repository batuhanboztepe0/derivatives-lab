"""
data/fetcher.py
===============
Cache-first market-data fetch for the real-data verification notebooks.

Live yfinance pulls are not reproducible — prices move, Yahoo rate-limits, and CI
has no network — so every fetch is pinned to a dated parquet under data/cache/ and
re-read from there on later runs.  After the first successful fetch the notebook
is offline and deterministic; CI never touches the network (it runs only tests/).

`fetch_and_cache` is a thin memoiser: it owns the cache path and the read/write,
and takes the actual fetch as a closure, so it stays agnostic to *what* is fetched
(option chains, price history, VIX) — each notebook supplies its own fetch_fn.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def fetch_and_cache(
    ticker: str,
    what: str,
    date: str,
    fetch_fn: Callable[[], pd.DataFrame],
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Return a dated, cached DataFrame; run fetch_fn() only on a cache miss.

    Parameters
    ----------
    ticker   : Underlying symbol, e.g. "SPY" (cache-key component).
    what     : Short label for the data kind, e.g. "vol_surface" (cache-key).
    date     : Snapshot date "YYYY-MM-DD" pinned into the filename — the notebook
               passes a fixed string so every re-run hits the same cache.
    fetch_fn : Zero-arg closure doing the live yfinance call on a miss; returns a
               DataFrame.
    cache_dir: Where to read/write (default data/cache/, gitignored).

    Cache file: data/cache/<ticker>_<what>_<date>.parquet.
    """
    path = Path(cache_dir) / f"{ticker}_{what}_{date}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
