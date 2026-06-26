"""
ml/vol_surface_nn.py
====================
Neural network for volatility surface fitting and interpolation.

THE PROBLEM
-----------
Black-Scholes assumes σ is constant across all strike prices.
Reality: IV varies with strike (smile/skew) and expiry (term structure).

If you plot IV against moneyness (K/S) and time-to-expiry (T):
  - You get a curved 3D surface, not a flat plane
  - The shape reveals what the market REALLY thinks about tail risk

SMILE vs SKEW
-------------
Smile: IV high on BOTH sides of ATM (symmetric U-shape).
       Common in FX — crashes can go either direction.

Skew (put skew): IV high on LEFT side only (deep OTM puts expensive).
       Common in equities — markets fall fast, rise slow.
       Deep OTM puts = crash insurance → high demand → high IV.
       This is the market saying: "BS log-normal is wrong, left tail is fatter."

WHY NEURAL NETWORK?
-------------------
The vol surface is nonlinear and changes daily.
Parametric models (SVI, SABR) fit well but have rigid shapes.
A neural net learns the mapping:

    f(moneyness, time_to_expiry) → IV

directly from market data, capturing any shape without assumptions.

WHY NOT LINEAR INTERPOLATION?
------------------------------
The surface is curved, especially:
  - Near ATM where IV changes rapidly
  - Near expiry where surface becomes steep
  - In smile/skew regions where nonlinearity is strong
Linear interpolation draws straight lines between points — misses curvature.
NN learns the nonlinear shape from all observed points simultaneously.

ARCHITECTURE
------------
Input  : [moneyness (K/S), log(T)]   (2 features)
Hidden : 3 × 64 neurons, ReLU, Dropout
Output : 1 scalar, predicted IV at that (K/S, T) point

log(T) instead of T because vol term structure is roughly log-linear.

DATA NOTE
---------
The bundled demo (run as __main__) fits this network to a hand-specified synthetic
surface from simulate_vol_surface(). The real-data path fetch_vol_surface() pulls a
live option chain, but VolSurfaceNN is demonstrated on the synthetic surface here,
not fit to market quotes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SIGMA_MAX, SIGMA_MIN
from models.black_scholes import BlackScholes

# ------------------------------------------------------------------
# Data: fetch real vol surface from yfinance
# ------------------------------------------------------------------

def fetch_vol_surface(ticker: str = "AAPL", r: float = 0.0438) -> pd.DataFrame:
    """
    Fetch real implied volatility surface from option chain.

    For each (strike, expiry) with valid market price:
      - Compute moneyness = K / S
      - Compute T = days to expiry / 365
      - Solve for IV using BlackScholes.implied_vol()

    Returns DataFrame: [moneyness, T, IV, strike, expiry]
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance") from None

    from datetime import datetime

    stock   = yf.Ticker(ticker)
    S       = float(stock.history(period="1d")["Close"].iloc[-1])
    records = []

    for expiry in stock.options:
        try:
            chain = stock.option_chain(expiry).calls   # calls only; OTM puts (where the left skew lives) are not fetched
            chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0) & (chain["volume"] > 10)]
            T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365

            if T <= 0:
                continue

            for _, row in chain.iterrows():
                K           = float(row["strike"])
                market_price = (float(row["bid"]) + float(row["ask"])) / 2
                moneyness   = K / S

                # Filter to reasonable moneyness range
                if not (0.7 <= moneyness <= 1.3):
                    continue

                bs = BlackScholes(S, K, T, r, sigma=0.2)
                iv = bs.implied_vol(market_price, "call")

                if not np.isnan(iv) and 0.01 < iv < 2.0:
                    records.append({
                        "moneyness": moneyness,
                        "T":         T,
                        "log_T":     np.log(T),
                        "IV":        iv,
                        "strike":    K,
                        "expiry":    expiry,
                        "S":         S,
                    })
        except Exception:
            continue

    return pd.DataFrame(records)


def simulate_vol_surface(n_points: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic vol surface when live data is unavailable.

    Uses a hand-specified skew + smile + term-structure shape (not a calibrated SVI):
      IV(k, T) = ATM_vol + skew×k + smile×k² + term_structure×√T
    where k = log(moneyness) = log(K/S)

    This captures:
      - Left skew (put premium for OTM puts)
      - Smile curvature
      - Vol term structure (higher vol for shorter expiries)
    """
    rng = np.random.default_rng(seed)

    moneyness = rng.uniform(0.75, 1.25, n_points)
    T         = rng.uniform(0.05, 1.0, n_points)
    k         = np.log(moneyness)               # log-moneyness

    # hand-specified surface shape
    atm_vol       = 0.20
    skew          = -0.15    # negative = put skew (left side higher)
    smile         =  0.30    # curvature
    term_structure= -0.05    # shorter expiry = higher vol

    IV = (atm_vol
          + skew * k
          + smile * k**2
          + term_structure * np.sqrt(T)
          + rng.normal(0, 0.005, n_points))     # small noise

    IV = np.clip(IV, 0.05, 1.5)

    return pd.DataFrame({
        "moneyness": moneyness,
        "T":         T,
        "log_T":     np.log(T),
        "IV":        IV,
    })


# ------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------

class VolSurfaceNN:
    """
    Neural network that learns the mapping:
        (moneyness, log_T) → IV

    Architecture: 3 hidden layers × 64 neurons
    Activation: ReLU (nonlinear, no vanishing gradient)
    Regularisation: Dropout(0.1) — prevents overfitting to noise in surface
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 3,
                 epochs: int = 200, lr: float = 1e-3, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.lr          = lr
        self.dropout     = dropout
        self.model       = None
        self.scaler_X    = None
        self.scaler_y    = None
        self.fitted      = False
        self.train_losses = []

    def fit(self, df: pd.DataFrame) -> VolSurfaceNN:
        """
        Train on observed (moneyness, T, IV) triplets.

        Features: [moneyness, log_T]
        Target:   IV
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("pip install torch") from None

        from sklearn.preprocessing import StandardScaler

        X = df[["moneyness", "log_T"]].values.astype(np.float32)
        y = df["IV"].values.astype(np.float32).reshape(-1, 1)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X).astype(np.float32)
        y_scaled = self.scaler_y.fit_transform(y).astype(np.float32)

        X_t = torch.FloatTensor(X_scaled)
        y_t = torch.FloatTensor(y_scaled)

        # Build network dynamically
        layers = []
        in_size = 2
        for _ in range(self.num_layers):
            layers += [
                nn.Linear(in_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_size = self.hidden_size
        layers.append(nn.Linear(self.hidden_size, 1))
        self.model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
            self.train_losses.append(loss.item())
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} — Loss: {loss.item():.6f}")

        self.fitted = True
        return self

    def predict(self, moneyness: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Predict IV for arbitrary (moneyness, T) combinations.
        This is the interpolation — works anywhere on the surface,
        not just observed market strikes.
        """
        if not self.fitted:
            raise RuntimeError("Fit model first: .fit(df)")

        import torch

        log_T   = np.log(np.maximum(T, 1e-6))
        X       = np.column_stack([moneyness, log_T]).astype(np.float32)
        X_scaled = self.scaler_X.transform(X).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(torch.FloatTensor(X_scaled)).numpy()

        iv = self.scaler_y.inverse_transform(preds_scaled).ravel()
        return np.clip(iv, SIGMA_MIN, SIGMA_MAX)

    def plot_surface(self, title: str = "Learned Volatility Surface") -> None:
        """3D plot of the learned surface vs observed data."""
        if not self.fitted:
            raise RuntimeError("Fit model first")

        m_grid = np.linspace(0.75, 1.25, 50)
        T_grid = np.linspace(0.05, 1.0,  50)
        M, T   = np.meshgrid(m_grid, T_grid)

        IV_pred = self.predict(M.ravel(), T.ravel()).reshape(M.shape)

        fig = plt.figure(figsize=(12, 7))
        ax  = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(M, T, IV_pred * 100,
                               cmap="viridis", alpha=0.8, edgecolor="none")

        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Time to Expiry (years)")
        ax.set_zlabel("Implied Volatility (%)")
        ax.set_title(title)
        fig.colorbar(surf, shrink=0.4, label="IV (%)")
        plt.tight_layout()
        plt.show()

    def plot_smile(self, T_values: tuple = (0.1, 0.25, 0.5, 1.0)) -> None:
        """
        Plot volatility smile/skew at fixed expiry dates.
        Shows how IV varies with moneyness at different maturities.
        """
        if not self.fitted:
            raise RuntimeError("Fit model first")

        moneyness_grid = np.linspace(0.75, 1.25, 100)
        plt.figure(figsize=(10, 6))

        for T_val in T_values:
            T_arr  = np.full_like(moneyness_grid, T_val)
            iv_arr = self.predict(moneyness_grid, T_arr) * 100
            plt.plot(moneyness_grid, iv_arr, linewidth=2, label=f"T = {T_val:.2f}y")

        plt.axvline(1.0, color="black", linewidth=0.8, linestyle="--", label="ATM")
        plt.xlabel("Moneyness (K/S)")
        plt.ylabel("Implied Volatility (%)")
        plt.title("Volatility Smile/Skew at Different Maturities")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# Quick run
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating simulated vol surface...")
    df = simulate_vol_surface(n_points=1000)
    print(f"Surface points: {len(df)}")
    print(df[["moneyness", "T", "IV"]].describe())

    print("\nTraining neural network...")
    nn = VolSurfaceNN(epochs=200, hidden_size=64)
    nn.fit(df)

    # Test interpolation at new points
    test_m = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
    test_T = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
    preds  = nn.predict(test_m, test_T)
    print("\nInterpolation test (T=0.25):")
    for m, iv in zip(test_m, preds, strict=False):
        print(f"  Moneyness {m:.2f} → IV {iv:.2%}")

    nn.plot_smile()
    nn.plot_surface()
