"""
ml/deep_hedging.py
==================
Differentiable deep hedging (Bühler et al. 2019) in pure PyTorch.

THE IDEA
--------
Black-Scholes says a sold option can be replicated by continuously holding
Δ_BS(S,t) shares — in the frictionless, continuous-time limit the hedge is
perfect and the hedging error is exactly zero.  Reality breaks both
assumptions: you rebalance in discrete time and you pay transaction costs.
Under those frictions the variance-minimising hedge is *no longer* the BS
delta, and there is no closed form for the optimal strategy.

Deep hedging learns it directly.  We parametrise the hedge ratio as a small
neural network π(state) → holding, simulate option-and-hedge P&L over Monte
Carlo price paths, and minimise a convex risk measure of the terminal P&L by
backpropagating through the entire simulated trajectory (backprop-through-time).
No environment, no replay buffer, no policy-gradient estimator — the simulator
itself is differentiable, so we get exact gradients of the risk w.r.t. the
network weights.

P&L ACCOUNTING (discounted to t=0, short one option)
----------------------------------------------------
Work in discounted prices X_t = e^{-rt}·S_t, a martingale under Q.  A
self-financing strategy holding δ_t shares over [t, t+1] earns discounted
trading gains Σ_t δ_t·(X_{t+1} − X_t).  Selling the option for its premium p0
and paying it back at expiry gives

    PnL = p0 + Σ_t δ_t·(X_{t+1} − X_t) − e^{−rT}·payoff(S_T) − cost

    cost = c · [ Σ_t |δ_t − δ_{t−1}|·X_t  +  |δ_{n−1}|·X_T ]   (δ_{−1}=0)

In discounted units perfect replication means PnL ≡ 0, so a *good* hedge drives
the whole PnL distribution to a tight spike at zero.

THE CRITICAL SANITY CHECK
-------------------------
With c = 0 and many steps the learned policy must converge to the Black-Scholes
delta — anything else is a bug.  Turn on transaction costs and the optimum
pulls *away* from BS delta into a no-trade band: the policy stops chasing every
small move because re-hedging is no longer free.  That gap is the whole point —
it is what a closed-form delta cannot tell you.

RISK MEASURES
-------------
- "mean_var"  : −E[PnL] + ½·λ·Var(PnL).  Simplest; its frictionless optimum is
                the variance-minimising hedge ≈ BS delta.
- "entropic"  : (1/λ)·log E[exp(−λ·PnL)] (exponential utility / entropic risk).
- "cvar"      : CVaR_α of the loss via Rockafellar-Uryasev, with the VaR level
                carried as a jointly-optimised scalar.  Focuses capital on the
                worst α tail — the risk a desk actually cares about.

WHERE THIS BREAKS DOWN
----------------------
- The learned policy is only as good as the simulator: GBM paths here mean the
  agent never sees jumps or stochastic vol unless you feed it those paths.
- Training is stochastic; results are seeded (config.SEED) but not bit-exact
  across hardware.
- No market impact / no discrete lot sizes — costs are proportional only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import DEFAULT_TRANSACTION_COST, SEED
from models.black_scholes import BlackScholes

OptionType = Literal["call", "put"]
RiskMeasure = Literal["cvar", "entropic", "mean_var"]


@dataclass
class DeepHedger:
    """
    Differentiable deep hedger for a single European option, short one unit.

    Parameters
    ----------
    S0, K, T, r, sigma : Market parameters of the option being hedged.
    option_type   : "call" or "put" (the instrument we are short).
    n_steps       : Number of discrete rebalancing dates.
    tc            : Proportional transaction cost (config.DEFAULT_TRANSACTION_COST).
    risk          : Convex risk measure to minimise — see module docstring.
    risk_aversion : λ for "mean_var"/"entropic" (ignored by "cvar").
    cvar_alpha    : Tail level α for "cvar" (e.g. 0.5 = mean of worst 50%).
    hidden_size   : Width of the two-hidden-layer policy MLP.
    seed          : Seed for path simulation and weight init (reproducibility).
    hedge_options : Optional list of (type, strike) European options the policy may
                    hold *statically* alongside the dynamic share hedge — e.g.
                    [("put", 90), ("call", 110)] (a long strangle).  Each is bought
                    once at t=0 at its fair price (the mean discounted payoff over the
                    training paths, so the overlay is zero-NPV by construction) and
                    held to T.  This is the only way to hedge the convex jump risk
                    that a delta-only hedge cannot touch.  None → pure delta hedging.
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = "call"
    n_steps: int = 30
    tc: float = DEFAULT_TRANSACTION_COST
    risk: RiskMeasure = "cvar"
    risk_aversion: float = 1.0
    cvar_alpha: float = 0.5
    hidden_size: int = 32
    seed: int = SEED
    hedge_options: list[tuple[OptionType, float]] | None = None

    def __post_init__(self) -> None:
        self._policy = None         # nn.Sequential, built in fit()
        self._cvar_w = None         # nn.Parameter, the VaR level for CVaR
        self._opt_qty = None        # nn.Parameter, learned hedge-option quantities
        self._hedge_prices = None   # fair prices of the hedge options (Tensor)
        self.fitted = False
        self.train_losses: list[float] = []

    @property
    def premium(self) -> float:
        """Sale price of the option = Black-Scholes price at t=0."""
        return BlackScholes(self.S0, self.K, self.T, self.r, self.sigma).price(self.option_type)

    # ── Path simulation (risk-neutral GBM) ────────────────────────────────────

    def simulate_paths(self, n_paths: int, seed: int | None = None):
        """
        Simulate `n_paths` GBM price paths under Q; shape (n_paths, n_steps+1).

        Pass `seed` for a reproducible evaluation set; leave it None during
        training so each epoch sees a fresh, independent batch (drawing from the
        global RNG, which fit() seeds once at the start).
        """
        import torch

        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        dt = self.T / self.n_steps
        Z = torch.randn(n_paths, self.n_steps, generator=gen)
        log_ret = (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * Z
        log_path = torch.cumsum(log_ret, dim=1)
        log_path = torch.cat([torch.zeros(n_paths, 1), log_path], dim=1)
        return self.S0 * torch.exp(log_path)

    # ── Policy network and rollout ────────────────────────────────────────────

    def _build_policy(self, nn):
        """Two-hidden-layer MLP: [log-moneyness, τ/T, current holding] → holding."""
        return nn.Sequential(
            nn.Linear(3, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def _policy_holdings(self, torch, paths):
        """
        Roll the policy forward, feeding its previous holding back as state.

        Threading the current holding into the state is what lets the network
        learn a *path-dependent* no-trade band when costs are on.  Returns the
        (n_paths, n_steps) matrix of holdings δ_t.
        """
        n_paths = paths.shape[0]
        dt = self.T / self.n_steps
        h_prev = torch.zeros(n_paths)
        cols = []
        for t in range(self.n_steps):
            S_t = paths[:, t]
            tau = self.T - t * dt
            feat = torch.stack([
                torch.log(S_t / self.K),
                torch.full((n_paths,), tau / self.T),
                h_prev,
            ], dim=1)
            h_t = self._policy(feat).squeeze(-1)
            cols.append(h_t)
            h_prev = h_t
        return torch.stack(cols, dim=1)

    def _bs_delta_holdings(self, torch, paths):
        """Black-Scholes delta at each (S_t, τ) — the frictionless benchmark hedge."""
        dt = self.T / self.n_steps
        cols = []
        for t in range(self.n_steps):
            tau = max(self.T - t * dt, 1e-12)
            d1 = (torch.log(paths[:, t] / self.K)
                  + (self.r + 0.5 * self.sigma ** 2) * tau) / (self.sigma * math.sqrt(tau))
            delta = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))   # N(d1)
            if self.option_type == "put":
                delta = delta - 1.0
            cols.append(delta)
        return torch.stack(cols, dim=1)

    # ── Differentiable P&L engine (shared by policy and benchmark) ────────────

    def _option_overlay(self, torch, paths, opt_qty):
        """
        Discounted P&L of a static long option position, net of its fair cost.

        Each option j is held in quantity opt_qty[j], pays e^{-rT}·payoff_j(S_T),
        and was bought at self._hedge_prices[j] (its fair value), so the leg is
        zero-mean under the path measure and acts purely as a risk (gamma) hedge.
        """
        S_T = paths[:, -1]
        disc = math.exp(-self.r * self.T)
        overlay = torch.zeros(paths.shape[0])
        for j, (otype, strike) in enumerate(self.hedge_options):
            if otype == "call":
                po = torch.clamp(S_T - strike, min=0.0)
            else:
                po = torch.clamp(strike - S_T, min=0.0)
            overlay = overlay + opt_qty[j] * (disc * po - self._hedge_prices[j])
        return overlay

    def _pnl(self, torch, paths, holdings, opt_qty=None):
        """
        Discounted terminal P&L for short-one-option, given a holdings matrix.

        Identical accounting for the learned policy and the BS-delta benchmark,
        so any P&L difference is purely the hedge, not the bookkeeping.  When
        opt_qty is supplied the static option overlay (gamma hedge) is added.
        """
        n = self.n_steps
        dt = self.T / n
        disc = torch.exp(-self.r * torch.arange(n + 1, dtype=paths.dtype) * dt)
        X = paths * disc                              # discounted prices
        dX = X[:, 1:] - X[:, :-1]
        hedge_gain = (holdings * dX).sum(dim=1)

        # Transaction cost: every rebalance (δ_{-1}=0) plus the terminal unwind.
        prev = torch.cat([torch.zeros(paths.shape[0], 1), holdings[:, :-1]], dim=1)
        turnover = ((holdings - prev).abs() * X[:, :-1]).sum(dim=1)
        turnover = turnover + holdings[:, -1].abs() * X[:, -1]
        cost = self.tc * turnover

        S_T = paths[:, -1]
        if self.option_type == "call":
            payoff = torch.clamp(S_T - self.K, min=0.0)
        else:
            payoff = torch.clamp(self.K - S_T, min=0.0)
        disc_payoff = math.exp(-self.r * self.T) * payoff

        pnl = self.premium + hedge_gain - disc_payoff - cost
        if opt_qty is not None:
            pnl = pnl + self._option_overlay(torch, paths, opt_qty)
        return pnl

    def _risk(self, torch, pnl):
        """Convex risk measure of the P&L (lower is better)."""
        if self.risk == "mean_var":
            return -pnl.mean() + 0.5 * self.risk_aversion * pnl.var()
        if self.risk == "entropic":
            lam = self.risk_aversion
            n = pnl.shape[0]
            return (torch.logsumexp(-lam * pnl, dim=0) - math.log(n)) / lam
        # cvar — Rockafellar-Uryasev with jointly-optimised VaR level _cvar_w
        loss = -pnl
        return self._cvar_w + torch.clamp(loss - self._cvar_w, min=0.0).mean() / self.cvar_alpha

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, epochs: int = 300, batch_size: int = 4096, lr: float = 1e-3,
            verbose: bool = False, paths_fn=None) -> DeepHedger:
        """
        Train the hedging policy by minimising the risk measure over fresh Monte
        Carlo batches (backprop-through-time).  Returns self.

        paths_fn : optional callable(batch_size) -> Tensor (batch, n_steps+1).
                   Supply this to train in a non-GBM "world" — e.g. Merton-jump or
                   Heston paths from the model zoo — so the policy learns to hedge
                   dynamics that the BS-delta benchmark cannot.  Defaults to the
                   built-in risk-neutral GBM simulator.
        """
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self._policy = self._build_policy(nn)
        params = list(self._policy.parameters())
        if self.risk == "cvar":
            self._cvar_w = nn.Parameter(torch.zeros(()))
            params.append(self._cvar_w)

        draw = paths_fn if paths_fn is not None else self.simulate_paths

        # Static option overlay: price each hedge option fairly off the path measure
        # (mean discounted payoff), then learn its quantity jointly with the policy.
        if self.hedge_options:
            with torch.no_grad():
                price_batch = draw(max(batch_size, 20_000))
                S_T = price_batch[:, -1]
                disc = math.exp(-self.r * self.T)
                prices = []
                for otype, strike in self.hedge_options:
                    po = (torch.clamp(S_T - strike, min=0.0) if otype == "call"
                          else torch.clamp(strike - S_T, min=0.0))
                    prices.append(disc * po.mean())
                self._hedge_prices = torch.stack(prices)
            self._opt_qty = nn.Parameter(torch.zeros(len(self.hedge_options)))
            params.append(self._opt_qty)

        opt = torch.optim.Adam(params, lr=lr)
        self.train_losses = []
        for epoch in range(epochs):
            paths = draw(batch_size)                          # fresh batch each epoch
            opt.zero_grad()
            pnl = self._pnl(torch, paths, self._policy_holdings(torch, paths), opt_qty=self._opt_qty)
            loss = self._risk(torch, pnl)
            loss.backward()
            opt.step()
            self.train_losses.append(float(loss.item()))
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  epoch {epoch + 1}/{epochs}  risk={loss.item():.5f}")

        self.fitted = True
        return self

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Call .fit() before evaluating the policy.")

    def policy_pnl(self, paths=None, n_paths: int = 50_000, seed: int = SEED) -> np.ndarray:
        """Terminal P&L distribution of the learned policy (numpy, no grad)."""
        import torch

        self._require_fitted()
        if paths is None:
            paths = self.simulate_paths(n_paths, seed=seed)
        with torch.no_grad():
            holdings = self._policy_holdings(torch, paths)
            return self._pnl(torch, paths, holdings, opt_qty=self._opt_qty).numpy()

    def bs_delta_pnl(self, paths=None, n_paths: int = 50_000, seed: int = SEED) -> np.ndarray:
        """Terminal P&L of the Black-Scholes delta benchmark on the same accounting."""
        import torch

        if paths is None:
            paths = self.simulate_paths(n_paths, seed=seed)
        with torch.no_grad():
            return self._pnl(torch, paths, self._bs_delta_holdings(torch, paths)).numpy()

    def option_quantities(self) -> dict[str, float]:
        """Learned static quantity held in each hedge option ({} if none)."""
        if not self.hedge_options or self._opt_qty is None:
            return {}
        q = self._opt_qty.detach().numpy()
        return {f"{ot}@{k:g}": float(q[j]) for j, (ot, k) in enumerate(self.hedge_options)}

    def policy_holdings(self, spot, t_step: int = 0, prev_holding=None) -> np.ndarray:
        """
        Learned holding across a grid of spot prices at a given rebalancing step.

        Used to plot π against the BS-delta curve.  `prev_holding` defaults to the
        BS delta at each spot, so the plot shows where the policy *would* move from
        the benchmark — the no-trade band is visible as a flat region around it.
        """
        import torch

        self._require_fitted()
        spot = np.atleast_1d(np.asarray(spot, dtype=np.float64))
        tau = max(self.T - t_step * (self.T / self.n_steps), 1e-12)
        if prev_holding is None:
            prev_holding = self.bs_delta(spot, tau)
        prev = np.broadcast_to(prev_holding, spot.shape).astype(np.float32)
        feat = torch.stack([
            torch.tensor(np.log(spot / self.K), dtype=torch.float32),
            torch.full((spot.size,), tau / self.T),
            torch.tensor(prev),
        ], dim=1)
        with torch.no_grad():
            return self._policy(feat).squeeze(-1).numpy()

    def bs_delta(self, spot, tau: float) -> np.ndarray:
        """Black-Scholes delta as a numpy array (benchmark curve for plots)."""
        spot = np.atleast_1d(np.asarray(spot, dtype=np.float64))
        tau = max(tau, 1e-12)
        d1 = (np.log(spot / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau) / (
            self.sigma * math.sqrt(tau))
        from scipy.stats import norm
        delta = norm.cdf(d1)
        return delta - 1.0 if self.option_type == "put" else delta
