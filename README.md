# derivatives-lab

Quantitative derivatives research lab. Three threads:

1. **Pricing model zoo** — Black-Scholes, Heston (stochastic vol), Monte Carlo, local vol — compared on pricing accuracy, smile/skew reproduction, and speed.
2. **RL deep hedging** — a PyTorch agent that learns a hedging policy under transaction costs and discrete time, benchmarked against Black-Scholes delta-hedging.
3. **Prediction markets as a derivative** — treating a binary prediction-market contract as a digital option, through the Black-Scholes / PDE (and bounded-martingale) lens.

> Build roadmap and design spec: see [`PLAN.md`](PLAN.md). Start at **Faz A**.

## Layout

```
models/        # pricers: black_scholes, heston, monte_carlo, local_vol (+ pde_solver TBD)
ml/            # vol_surface_nn (+ deep_hedging TBD)
backtesting/   # metrics (sharpe, sortino, max drawdown, …)
research/      # notebooks: 01 model zoo · 02 deep hedging · 03 prediction markets
reports/       # written reports
tests/         # analytic-check unit tests
config.py      # seeds, default rate, tolerances, transaction cost
```

## Setup

```bash
make install          # pip install -e ".[all]"   (numpy/scipy/pandas/matplotlib/yfinance + sklearn/torch)
make ci               # ruff + mypy + pytest
```

Python ≥ 3.10. Heavy deps (`torch`, `scikit-learn`) live in the `[ml]` extra.

## Status

Core pricers (BS / Heston / Monte Carlo) and performance metrics migrated and working. `local_vol.py` is a stub to be implemented; `pde_solver.py`, `deep_hedging.py`, and the research notebooks are to be built per [`PLAN.md`](PLAN.md).
