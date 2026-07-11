"""Feature engineering for the demand forecaster.

All features are derivable at prediction time from the timestamp, recent
history, and the exogenous temperature input. Lag features use only past
values, so there is no target leakage: forecasting demand at time t uses
demand up to t-1 at the earliest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags (in hours) chosen to capture short-term persistence (1-3h), the daily
# cycle (24h), and the weekly cycle (168h).
LAG_HOURS = [1, 2, 3, 24, 168]
TARGET = "demand_mw"

FEATURE_COLUMNS = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
     "is_weekend", "temperature_c"]
    + [f"lag_{h}h" for h in LAG_HOURS]
    + ["rolling_24h_mean"]
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the model feature matrix from a raw demand DataFrame.

    Args:
        df: DataFrame indexed by hourly timestamps with columns
            `demand_mw` and `temperature_c`.

    Returns:
        DataFrame containing FEATURE_COLUMNS plus the target, with the
        initial rows (which lack full lag history) dropped.
    """
    out = pd.DataFrame(index=df.index)

    # Cyclical encodings: sin/cos pairs avoid the artificial discontinuity of
    # raw hour-of-day (23 -> 0) that tree models would otherwise have to
    # learn around.
    hours = df.index.hour
    dow = df.index.dayofweek
    doy = df.index.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    out["is_weekend"] = (dow >= 5).astype(int)

    out["temperature_c"] = df["temperature_c"]

    # Autoregressive features: strictly past values only.
    for h in LAG_HOURS:
        out[f"lag_{h}h"] = df[TARGET].shift(h)
    out["rolling_24h_mean"] = df[TARGET].shift(1).rolling(24).mean()

    out[TARGET] = df[TARGET]
    return out.dropna()
