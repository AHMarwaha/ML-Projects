"""Synthetic energy-demand data generation.

Generates an hourly electricity-demand series with the structure real demand
data exhibits: daily and weekly seasonality, an annual cycle, a mild upward
trend, temperature dependence, and heteroscedastic noise. Using synthetic
data keeps the repository self-contained and reproducible; the pipeline is
data-source agnostic, and the README documents how to point it at a real
feed (e.g. ENTSO-E) instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demand(
    start: str = "2023-01-01",
    periods: int = 24 * 365 * 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an hourly demand series with realistic seasonal structure.

    Args:
        start: First timestamp of the series.
        periods: Number of hourly observations (default: two years).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame indexed by timestamp with columns:
        - demand_mw: the target variable.
        - temperature_c: an exogenous driver (correlated with demand).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq="h")

    hours = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    t = np.arange(periods)

    # Seasonal components. Amplitudes are in megawatts.
    daily = 300 * np.sin(2 * np.pi * (hours - 7) / 24)        # morning/evening swing
    weekly = -150 * (dow >= 5)                                 # weekend dip
    annual = 200 * np.cos(2 * np.pi * (doy - 15) / 365)        # winter peak
    trend = 0.01 * t                                           # slow demand growth

    # Temperature: annual cycle + noise; demand rises when it is cold.
    temperature = 12 - 10 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2, periods)
    temp_effect = 15 * np.clip(15 - temperature, 0, None)

    base = 2000.0
    noise = rng.normal(0, 40 + 20 * (dow >= 5), periods)       # noisier weekends

    demand = base + daily + weekly + annual + trend + temp_effect + noise

    return pd.DataFrame(
        {"demand_mw": demand, "temperature_c": temperature}, index=idx
    ).rename_axis("timestamp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demand data")
    parser.add_argument("--out", type=Path, default=Path("data/demand.csv"))
    parser.add_argument("--periods", type=int, default=24 * 365 * 2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demand(periods=args.periods, seed=args.seed)
    df.to_csv(args.out)
    print(f"Wrote {len(df)} rows to {args.out}")
