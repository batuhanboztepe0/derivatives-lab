"""
config.py
=========
Centralised configuration for options-pricing-lab.

All random seeds, numeric tolerances, and path constants live here.
Import from this module instead of hardcoding values anywhere else.

Usage
-----
from config import SEED, TRADING_DAYS, DATA_DIR
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────
SEED: int = 42          # global random seed — used everywhere

# ── Calendar ─────────────────────────────────────────────────────
TRADING_DAYS: int = 252   # annualisation factor

# ── Numeric tolerances ───────────────────────────────────────────
T_MIN:     float = 1e-6   # minimum time-to-expiry (years)
SIGMA_MIN: float = 1e-6   # minimum implied vol
SIGMA_MAX: float = 10.0   # maximum implied vol for IV solver bracket

# ── Default market parameters ────────────────────────────────────
DEFAULT_RISK_FREE_RATE: float = 0.0438   # ~current Fed Funds (update periodically)
DEFAULT_DIVIDEND_YIELD: float = 0.0

# ── Data ─────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = ROOT_DIR / "data" / "cache"
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass   # tolerate a read-only filesystem at import; cache writes will surface their own error

# ── Heston calibration defaults ──────────────────────────────────
HESTON_DE_POPSIZE: int   = 15
HESTON_DE_MAXITER: int   = 300
HESTON_FELLER_PENALTY: float = 10.0

# ── Backtesting ──────────────────────────────────────────────────
HEDGE_REBALANCE_FREQ: str = "1D"   # daily rebalancing
DEFAULT_TRANSACTION_COST: float = 0.0005   # 5 bps per trade
