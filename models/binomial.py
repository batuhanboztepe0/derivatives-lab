"""
models/binomial.py
==================
Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

WHY A TREE?
-----------
Black-Scholes gives a closed form only for European exercise.  The binomial
tree is the most transparent numerical alternative: it discretises the stock
into a recombining lattice of up/down moves and prices *backward* from the
known terminal payoff.  Two things fall out almost for free:

  1. **Convergence to Black-Scholes.**  As the number of steps N → ∞, the
     discrete CRR price converges to the continuous BS price.  This is the
     classic pedagogical bridge between discrete and continuous finance and
     the primary sanity check for this class.
  2. **American early exercise.**  At every node we can compare the value of
     holding (discounted continuation) against the value of exercising now
     (intrinsic).  Closed-form BS cannot do this; the tree handles it with a
     single `max(...)` per node.

THE CRR PARAMETRISATION
-----------------------
Over each step dt = T/N the stock moves up by a factor u or down by d, chosen
to match the log-return volatility σ:

    u = e^{ σ√dt },   d = 1/u   (recombining: an up-then-down returns to start)

Under the risk-neutral measure the up-probability is fixed by no-arbitrage so
that the discounted stock is a martingale:

    p = (e^{r·dt} − d) / (u − d)

The option value at a node is the discounted expectation over its two children:

    V = e^{−r·dt} · [ p·V_up + (1−p)·V_down ]

For 0 < p < 1 (guaranteed when dt is small enough that d < e^{r·dt} < u) the
tree is arbitrage-free and European prices satisfy put-call parity exactly,
for *any* N — a second, N-independent test anchor.

WHERE THIS BREAKS DOWN
----------------------
- Convergence is O(1/N) and *oscillatory*: the error wobbles in sign as N
  steps past each strike node, so doubling N does not cleanly halve the error.
  (Smoothing tricks — averaging N and N+1, or a Black-Scholes terminal layer —
  remove the wobble; we keep the plain CRR scheme for clarity.)
- A single constant σ means the tree, like Black-Scholes, cannot reproduce a
  vol smile (see models/local_vol.py, models/heston.py).
- Path-dependent payoffs (Asian, lookback) do not fit a recombining lattice;
  use Monte Carlo (see models/monte_carlo.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass
class BinomialTree:
    """
    Cox-Ross-Rubinstein binomial pricer for European and American options.

    Parameters
    ----------
    S     : Current stock price.
    K     : Strike price.
    T     : Time to maturity (years).
    r     : Risk-free rate (continuously compounded).
    sigma : Annualised volatility.
    N     : Number of time steps in the tree (more → closer to Black-Scholes).

    Examples
    --------
    >>> tree = BinomialTree(S=100, K=100, T=1.0, r=0.05, sigma=0.2, N=500)
    >>> round(tree.price("call", "european"), 2)
    10.45
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    N: int = 500

    def price(
        self,
        option_type: OptionType = "call",
        exercise: ExerciseStyle = "european",
    ) -> float:
        """
        Price a European or American option by backward induction on the CRR tree.

        Steps:
          1. Build the CRR up/down factors and the risk-neutral up-probability.
          2. Evaluate the terminal payoff across all N+1 leaf nodes.
          3. Roll back step by step: each node = discounted expectation of its
             two children; for American exercise, take the max with intrinsic.

        Returns
        -------
        float : Option price at t=0.
        """
        N = self.N
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-self.r * dt)
        p = (np.exp(self.r * dt) - d) / (u - d)   # risk-neutral up-probability

        # ── Terminal layer (step N): S_T at node j = S·u^j·d^(N-j), j = 0..N ──
        j = np.arange(N + 1)
        S_T = self.S * u ** j * d ** (N - j)
        if option_type == "call":
            V = np.maximum(S_T - self.K, 0.0)
        else:
            V = np.maximum(self.K - S_T, 0.0)

        # ── Roll back to the root ─────────────────────────────────────────────
        for i in range(N - 1, -1, -1):
            V = disc * (p * V[1:] + (1.0 - p) * V[:-1])   # discounted expectation
            if exercise == "american":
                j = np.arange(i + 1)
                S_i = self.S * u ** j * d ** (i - j)       # spot at each node of step i
                if option_type == "call":
                    intrinsic = np.maximum(S_i - self.K, 0.0)
                else:
                    intrinsic = np.maximum(self.K - S_i, 0.0)
                V = np.maximum(V, intrinsic)               # early-exercise decision

        return float(V[0])

    def early_exercise_premium(self, option_type: OptionType = "put") -> float:
        """
        American value minus European value — the worth of the early-exercise right.

        Always ≥ 0.  For a non-dividend call it is exactly 0 (never optimal to
        exercise early when r > 0); for an in-the-money put with r > 0 it is
        strictly positive.  A compact way to expose the one thing a tree can
        price that closed-form Black-Scholes cannot.
        """
        return self.price(option_type, "american") - self.price(option_type, "european")
