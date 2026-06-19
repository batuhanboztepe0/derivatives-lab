"""
models/pde_solver.py
====================
Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

THE BLACK-SCHOLES PDE
---------------------
Every hedged portfolio of a derivative and the underlying must earn the
risk-free rate (no-arbitrage).  Applying Itô's lemma to V(S,t) gives the
BS PDE (in calendar time):

    ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S − rV = 0

We price by solving *backward* in time from the known terminal payoff at
t = T toward t = 0.  With the substitution τ = T − t (time remaining), the
equation becomes a forward-in-τ diffusion, which is the standard form for
finite differences.  The grid lives on [0, S_max] × [0, T].

FINITE-DIFFERENCE DISCRETISATION
---------------------------------
Let V_i^n ≈ V(S_i, τ_n), where S_i = i·dS and τ_n = n·dt.
Uniform grids: dS = S_max / M,  dt = T / N.

For the spatial derivatives we use centred differences (2nd-order in dS):

    ∂²V/∂S² ≈ (V_{i+1} − 2V_i + V_{i-1}) / dS²
    ∂V/∂S   ≈ (V_{i+1} − V_{i-1}) / (2·dS)

The θ-scheme averages levels n and n+1 with weight θ (implicit) and
(1−θ) (explicit):

    (V_i^{n+1} − V_i^n) / dt = θ·L[V^{n+1}]_i + (1−θ)·L[V^n]_i

where L[V]_i = ½σ²S_i²(V_{i+1}−2V_i+V_{i-1})/dS² + rS_i(V_{i+1}−V_{i-1})/(2dS) − rV_i.

Define per-node coefficients (lowercase = scalar):

    a_i = ½·θ·( ½σ²S_i²/dS² − rS_i/(2dS) )   # sub-diagonal weight
    c_i = ½·θ·( ½σ²S_i²/dS² + rS_i/(2dS) )   # super-diagonal weight
    b_i = 1/dt + θ·( σ²S_i²/dS² + r )         # diagonal weight

The linear system at each time step is:
    −a_i·V_{i-1}^{n+1} + b_i·V_i^{n+1} − c_i·V_{i+1}^{n+1} = RHS_i

where the explicit RHS uses the same coefficients with (1−θ)/θ scaling.

WHY CRANK-NICOLSON IS THE RIGHT SCHEME
---------------------------------------
θ = 0  (fully explicit): conditionally stable, requires dt ≪ dS² — impractical
        for fine grids.  1st-order in time.

θ = 1  (fully implicit / backward Euler): unconditionally stable but only
        1st-order in time — diffuses features.

θ = ½ (Crank-Nicolson): unconditionally stable (von Neumann: amplification
        factor |(1−λ/2)/(1+λ/2)| ≤ 1 for real λ > 0).
        2nd-order in *both* time and space — O(dt² + dS²).

THE RANNACHER SMOOTHING FIX
-----------------------------
CN is 2nd-order on smooth payoffs but exhibits Gibbs-like oscillations when
the initial condition is discontinuous (digital payoffs).  The oscillations
arise because CN has no numerical dissipation.  Rannacher's fix: use θ=1
(fully implicit) for the first few backward steps, which is strongly
dissipative, then switch to CN.  Two implicit steps suffice to damp the
jump before CN loses its ability to dissipate.  Industry standard for
digital and barrier options.

BOUNDARY CONDITIONS
-------------------
At S = 0: diffusion and drift vanish; the PDE reduces to ∂V/∂τ = rV,
so V(0,τ) decays at rate e^{−rτ}.  We impose the exact Dirichlet BC.

At S = S_max: far from the strike, the call behaves like a forward contract;
the put and digital put probability → 0.  We impose exact time-dependent
Dirichlet BCs derived from limiting arguments.

WHERE THIS BREAKS DOWN
-----------------------
- Uniform S-grid places few points near the strike → log-uniform or
  concentrated grids give much better accuracy per node.
- American options require a free-boundary (early-exercise) constraint at
  each time step — not implemented here.
- Jump-diffusion models (Merton, Kou) need integro-differential extensions.
- Near expiry with ATM digital payoffs, even Rannacher+fine grids leave
  ~O(dS) error; call-spread approximations are used in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

OptionType = Literal["call", "put"]


@dataclass
class CrankNicolsonBS:
    """
    Crank-Nicolson finite-difference pricer for European BS options.

    Solves the Black-Scholes PDE backward in time on a uniform S-grid via a
    tridiagonal system at each step (scipy banded solver, O(M) per step).
    Supports vanilla and cash-or-nothing (digital) payoffs.

    Parameters
    ----------
    S       : Current stock price.
    K       : Strike price.
    T       : Time to maturity (years).
    r       : Risk-free rate.
    sigma   : Annualised volatility.
    S_max   : Upper boundary of the S-grid.  Default ~ max(4K, 4S).
              Should be large enough that the boundary condition error
              at S_max is negligible; 4× the ATM level is usually fine.
    M       : Number of spatial intervals (grid points = M+1).
    N       : Number of time steps.
    rannacher: Apply Rannacher smoothing — first 2 backward steps fully
              implicit (θ=1), then Crank-Nicolson (θ=0.5).  Strongly
              recommended for digital payoffs to kill Gibbs oscillations.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    S_max: float | None = None
    M: int = 200
    N: int = 200
    rannacher: bool = True

    def __post_init__(self) -> None:
        if self.S_max is None:
            self.S_max = max(4.0 * self.K, 4.0 * self.S)

    def price(
        self,
        option_type: OptionType = "call",
        payoff: Literal["vanilla", "digital"] = "vanilla",
    ) -> float:
        """
        Price a European option via Crank-Nicolson PDE solver.

        Steps:
          1. Build a uniform S-grid on [0, S_max].
          2. Set V at τ=0 (t=T) from the terminal payoff.
          3. March forward in τ (backward in t) using CN tridiagonal solves.
          4. Interpolate V(S, τ=T) at S = self.S.

        Parameters
        ----------
        option_type : "call" or "put".
        payoff      : "vanilla" (standard European) or "digital" (cash-or-nothing $1).

        Returns
        -------
        float : Option price at t=0.
        """
        S_max = self.S_max
        M, N = self.M, self.N
        r, sigma = self.r, self.sigma

        dS = S_max / M
        dt = self.T / N

        # ── S-grid ────────────────────────────────────────────────────────────
        S_grid = np.linspace(0.0, S_max, M + 1)   # (M+1,) including boundaries

        # ── Terminal payoff at τ=0 (t=T) ──────────────────────────────────────
        if payoff == "vanilla":
            if option_type == "call":
                V = np.maximum(S_grid - self.K, 0.0)
            else:
                V = np.maximum(self.K - S_grid, 0.0)
        else:  # digital (cash-or-nothing $1)
            if option_type == "call":
                V = (S_grid > self.K).astype(float)
            else:
                V = (S_grid < self.K).astype(float)
            # Half-jump at the node exactly at the strike (if any) reduces
            # the one-sided bias introduced by the step discontinuity.
            k_idx = np.where(np.isclose(S_grid, self.K))[0]
            if k_idx.size > 0:
                V[k_idx[0]] = 0.5

        # ── Per-node PDE coefficients for interior nodes i = 1..M-1 ──────────
        # (these are σ-independent of θ; θ rescales them at each step)
        S_i = S_grid[1:M]                   # interior S values, shape (M-1,)
        sig2S2 = sigma ** 2 * S_i ** 2      # σ²S²
        rS     = r * S_i                    # rS

        # Raw coefficients (before θ):
        #   A_i = ½ σ²S_i²/dS² − ½ rS_i/(2dS)   → sub-diagonal
        #   C_i = ½ σ²S_i²/dS² + ½ rS_i/(2dS)   → super-diagonal
        #   B_i = σ²S_i²/dS² + r                 → diagonal contribution
        A = 0.5 * (sig2S2 / dS ** 2 - rS / dS)
        C = 0.5 * (sig2S2 / dS ** 2 + rS / dS)
        B = sig2S2 / dS ** 2 + r

        def _bc(tau: float) -> tuple[float, float]:
            """Time-dependent Dirichlet boundary conditions at S=0 and S=S_max."""
            disc = np.exp(-r * tau)
            if payoff == "vanilla":
                if option_type == "call":
                    return 0.0, S_max - self.K * disc
                else:
                    return self.K * disc, 0.0
            else:  # digital
                if option_type == "call":
                    return 0.0, disc
                else:
                    return disc, 0.0

        # ── Time-stepping (forward in τ from 0 to T) ──────────────────────────
        V_int = V[1:M].copy()   # interior values only, shape (M-1,)

        for n_step in range(N):
            tau_curr = n_step * dt          # τ at start of this step
            tau_next = (n_step + 1) * dt    # τ after this step

            # Rannacher: first 2 steps fully implicit, then CN
            theta = 1.0 if (self.rannacher and n_step < 2) else 0.5
            theta1 = 1.0 - theta            # explicit weight

            bc0_curr, bcM_curr = _bc(tau_curr)
            bc0_next, bcM_next = _bc(tau_next)

            # ── Build implicit tridiagonal (LHS: A·V^{n+1} = rhs) ────────────
            # Diagonal:    1/dt + θ·B
            # Sub-diag:   -θ·A  (coefficient of V_{i-1})
            # Super-diag: -θ·C  (coefficient of V_{i+1})
            diag = 1.0 / dt + theta * B
            sub  = -theta * A               # length M-1, but only M-2 used (not first)
            sup  = -theta * C               # length M-1, but only M-2 used (not last)

            # scipy solve_banded uses (2,1) banded form:
            #   ab[0, j] = super-diagonal (row j, column j+1)  → ab[0, 1:]  = sup[:-1]
            #   ab[1, j] = diagonal                            → ab[1, :]   = diag
            #   ab[2, j] = sub-diagonal (row j, column j-1)   → ab[2, :-1] = sub[1:]
            ab = np.zeros((3, M - 1))
            ab[0, 1:]  = sup[:-1]
            ab[1, :]   = diag
            ab[2, :-1] = sub[1:]

            # ── Build explicit RHS ─────────────────────────────────────────────
            # RHS_i = (1/dt - θ1·B)·V_i + θ1·A·V_{i-1} + θ1·C·V_{i+1}
            #        + θ·A·bc0_next (first node) / θ·C·bcM_next (last node)
            #        + θ1·A·bc0_curr (first node) / θ1·C·bcM_curr (last node)
            rhs = (1.0 / dt - theta1 * B) * V_int

            # Interior contributions from neighbours
            rhs[1:]  += theta1 * A[1:]  * V_int[:-1]   # V_{i-1} for i >= 2
            rhs[:-1] += theta1 * C[:-1] * V_int[1:]    # V_{i+1} for i <= M-2

            # Boundary corrections — current level (explicit part)
            rhs[0]  += theta1 * A[0]  * bc0_curr
            rhs[-1] += theta1 * C[-1] * bcM_curr

            # Boundary corrections — next level (implicit part, moved to RHS)
            rhs[0]  += theta  * A[0]  * bc0_next
            rhs[-1] += theta  * C[-1] * bcM_next

            V_int = solve_banded((1, 1), ab, rhs)

        # ── Reconstruct full grid and interpolate at S = self.S ───────────────
        V_final = np.empty(M + 1)
        bc0_T, bcM_T = _bc(self.T)
        V_final[0]   = bc0_T
        V_final[M]   = bcM_T
        V_final[1:M] = V_int

        return float(np.interp(self.S, S_grid, V_final))
