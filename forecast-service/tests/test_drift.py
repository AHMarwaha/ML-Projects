"""Unit tests for drift detection: no false alarm on identical data, clear
detection on a shifted distribution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from drift import detect_drift, population_stability_index


def _frame(loc: float, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(loc, 1, n), "y": rng.normal(0, 1, n)})


def test_no_drift_on_same_distribution():
    ref, cur = _frame(0.0, seed=1), _frame(0.0, seed=2)
    report = detect_drift(ref, cur)
    assert report["drift_detected"] is False


def test_detects_mean_shift():
    ref, cur = _frame(0.0, seed=1), _frame(3.0, seed=2)
    report = detect_drift(ref, cur)
    assert report["drift_detected"] is True
    assert report["features"]["x"]["drifted"] is True


def test_psi_zero_for_identical_sample():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    assert population_stability_index(x, x) < 0.01
