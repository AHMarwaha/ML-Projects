"""Statistical data-drift detection.

Compares the feature distribution seen in production against the training
reference using two complementary tests per feature:

- Population Stability Index (PSI): a binned divergence measure. Industry
  rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major
  shift requiring action.
- Two-sample Kolmogorov-Smirnov test: nonparametric test of whether the two
  samples come from the same distribution.

A feature is flagged when PSI exceeds the threshold or the KS test rejects
at the configured significance level. The service exposes this through the
/drift endpoint; in a scheduled batch context the same function gates a
retraining job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.25
KS_ALPHA = 0.01


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and a current sample.

    Bin edges come from reference quantiles so every bin holds roughly equal
    reference mass, which keeps the statistic stable for skewed features.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    # Guard against duplicate edges from low-cardinality features.
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)

    # Avoid log(0): floor proportions at a small epsilon.
    ref_frac = np.clip(ref_frac, 1e-6, None)
    cur_frac = np.clip(cur_frac, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = PSI_THRESHOLD,
    ks_alpha: float = KS_ALPHA,
) -> dict:
    """Per-feature drift report over the columns shared by both frames.

    Returns:
        Dict with per-feature PSI, KS p-value, and a drifted flag, plus an
        overall `drift_detected` boolean.
    """
    report = {}
    for col in reference.columns:
        if col not in current.columns:
            continue
        ref, cur = reference[col].dropna().to_numpy(), current[col].dropna().to_numpy()
        if len(cur) < 30:
            # Too few samples for a meaningful test; skip rather than guess.
            continue
        psi = population_stability_index(ref, cur)
        ks_p = float(stats.ks_2samp(ref, cur).pvalue)
        report[col] = {
            "psi": round(psi, 4),
            "ks_pvalue": ks_p,
            "drifted": bool(psi > psi_threshold or ks_p < ks_alpha),
        }

    return {
        "features": report,
        "drift_detected": any(v["drifted"] for v in report.values()),
    }
