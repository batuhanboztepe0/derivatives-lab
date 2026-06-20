# derivatives-lab

A quantitative derivatives research lab: a zoo of option-pricing and hedging models, built
from scratch and tested with analytic anchors — then **stress-tested against real market
data**. Every load-bearing claim made in the synthetic world is verified (or honestly
qualified) on live SPY options, 10 years of SPY/VIX history, and 83k resolved Polymarket
markets.

## Highlights

The full write-up with numbers and caveats is in
**[`reports/MODEL_ZOO_FINDINGS.md`](reports/MODEL_ZOO_FINDINGS.md)**. In brief:

| # | Real-data verification | Result |
|---|---|---|
| V1 | Implied-vol smile (SPY chain) | Merton (jumps) fits the steep short-dated skew to **0.38 vol pts**; Heston underfits (**2.76**, structural); flat BS misses the wings (**6.16**). |
| V2 | Fat tails & jumps (10y daily SPY) | excess kurtosis **15.2**, left skew **−0.61**, Jarque–Bera rejects the Normal — GBM rejected. |
| V3 | Minimum-variance delta (SPY + VIX) | Large OOS hedging-variance reduction vs the synthetic `v₀=θ` baseline of −4% (magnitude inflated by the VIX-as-IV construction; the robust result is the **positive sign**, cf. Hull–White 2017 ~26% on real quotes). |
| V4 | Gamma P&L attribution | Short-gamma bleed concentrates on big moves: the top 5% move-days carry **38%** of the convexity P&L. |
| V5 | Deep hedger out-of-sample | On block-bootstrapped real returns, the cost-aware policy trades **~42% less** than BS-delta at comparable tail risk. |
| V6 | Favorite–longshot bias (83k Polymarket markets) | Longshots (p<0.10) priced **2.3%** resolve **1.6%**; favorites (p>0.90) priced **96%** resolve **99%**; slope **1.08**, robust across volume tiers and years. |

## Models

- **`models/`** — `black_scholes` (greeks, IV, digital options), `merton` (jump-diffusion,
  closed form + paths), `heston` (Carr–Madan FFT + quadrature + QE Monte Carlo + DE
  calibrator), `binomial` (CRR, American early exercise), `monte_carlo` (GBM, antithetic,
  exotics), `local_vol` (CEV + Dupire), `pde_solver` (Crank–Nicolson).
- **`ml/`** — `vol_surface_nn` (SVI-inspired surface fit + yfinance fetcher),
  `deep_hedging` (PyTorch policy, mean-variance / entropic / CVaR risk, transaction costs,
  external path sources, option overlay).
- **`backtesting/`** — Sharpe, Sortino, max drawdown, summary metrics.
- **`data/`** — `fetcher` (cache-first market-data pulls; cache is gitignored).
- **`research/`** — executed notebooks `04`–`09` (the V1–V6 verifications above).
- **`tests/`** — analytic-anchor unit tests.

## Setup

```bash
make install          # pip install -e ".[all]"   (numpy/scipy/pandas/matplotlib/yfinance + torch)
make ci               # ruff + mypy + pytest
```

Python ≥ 3.10. PyTorch lives in the `[ml]` extra; CI runs the test suite without it
(torch-dependent tests skip automatically).

## Tests & reproducibility

`pytest` — **98 tests**, anchored on analytic facts (λ=0 Merton collapses to Black–Scholes,
put–call parity, CRR → BS convergence, finite-difference vs closed-form deltas, discounted-
martingale checks). Real-data fetches are pinned to dated parquet caches so the notebooks
re-run offline and deterministically; CI runs only `tests/`, never the data notebooks.
