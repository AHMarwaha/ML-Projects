"""Unit tests for feature engineering: shape, leakage, and encoding checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from data import generate_demand
from features import FEATURE_COLUMNS, TARGET, build_features


def test_feature_columns_present():
    df = generate_demand(periods=24 * 30)
    feats = build_features(df)
    assert set(FEATURE_COLUMNS).issubset(feats.columns)
    assert TARGET in feats.columns


def test_no_nans_after_build():
    df = generate_demand(periods=24 * 30)
    feats = build_features(df)
    assert not feats.isna().any().any()


def test_lags_are_strictly_past():
    """lag_1h at time t must equal the raw target at t-1 (no leakage)."""
    df = generate_demand(periods=24 * 30)
    feats = build_features(df)
    t = feats.index[100]
    assert feats.loc[t, "lag_1h"] == df.loc[t - np.timedelta64(1, "h"), TARGET]


def test_cyclical_encoding_bounded():
    df = generate_demand(periods=24 * 30)
    feats = build_features(df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert feats[col].between(-1, 1).all()
