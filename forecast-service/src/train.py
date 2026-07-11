"""Train the demand forecaster with walk-forward evaluation and MLflow tracking.

The model is a histogram-based gradient-boosted regressor (scikit-learn),
which remains the strongest practical baseline for tabular forecasting.
Evaluation is walk-forward: the model is repeatedly trained on an expanding
historical window and scored on the period immediately after it, mirroring
how the model would actually be used in production. Random train/test splits
on time series leak future information and overstate accuracy.

MLflow logs parameters, per-fold metrics, and the final model artifact. If
MLflow is not installed the script still runs and prints metrics, so the
training path has no hard dependency on the tracking server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from data import generate_demand
from features import FEATURE_COLUMNS, TARGET, build_features

try:  # tracking is optional so tests and CI can run without an MLflow server
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False


def walk_forward_evaluate(
    features: pd.DataFrame,
    n_folds: int = 4,
    test_hours: int = 24 * 28,
    **model_params,
) -> tuple[list[dict], HistGradientBoostingRegressor]:
    """Expanding-window walk-forward evaluation.

    Fold i trains on all data up to a cut point and tests on the following
    `test_hours` hours. Returns per-fold metrics and a final model fit on
    the full dataset.
    """
    X = features[FEATURE_COLUMNS]
    y = features[TARGET]
    n = len(features)

    fold_metrics = []
    for fold in range(n_folds):
        test_end = n - (n_folds - 1 - fold) * test_hours
        test_start = test_end - test_hours
        if test_start <= test_hours:
            raise ValueError("Not enough data for the requested folds")

        model = HistGradientBoostingRegressor(**model_params)
        model.fit(X.iloc[:test_start], y.iloc[:test_start])
        preds = model.predict(X.iloc[test_start:test_end])
        actual = y.iloc[test_start:test_end]

        fold_metrics.append({
            "fold": fold,
            "mae": float(mean_absolute_error(actual, preds)),
            "mape": float(mean_absolute_percentage_error(actual, preds)),
        })

    final_model = HistGradientBoostingRegressor(**model_params)
    final_model.fit(X, y)
    return fold_metrics, final_model


def main(out_dir: Path, seed: int) -> None:
    model_params = {
        "max_iter": 300,
        "learning_rate": 0.08,
        "max_depth": 6,
        "early_stopping": True,
        "random_state": seed,
    }

    raw = generate_demand(seed=seed)
    features = build_features(raw)
    fold_metrics, model = walk_forward_evaluate(features, **model_params)

    mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
    mean_mape = float(np.mean([m["mape"] for m in fold_metrics]))
    print(f"Walk-forward MAE: {mean_mae:.1f} MW | MAPE: {mean_mape:.3%}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    # Persist the training feature distribution as the drift reference.
    reference = features[FEATURE_COLUMNS].describe().to_dict()
    (out_dir / "drift_reference.json").write_text(json.dumps(reference, indent=2))
    features[FEATURE_COLUMNS].sample(2000, random_state=seed).to_csv(
        out_dir / "drift_reference_sample.csv", index=False
    )
    print(f"Model and drift reference written to {out_dir}")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run():
            mlflow.log_params(model_params)
            mlflow.log_metric("walk_forward_mae", mean_mae)
            mlflow.log_metric("walk_forward_mape", mean_mape)
            for m in fold_metrics:
                mlflow.log_metric("fold_mae", m["mae"], step=m["fold"])
            mlflow.sklearn.log_model(model, name="model")
        print("Run logged to MLflow")
    else:
        print("MLflow not installed; skipped experiment tracking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the demand forecaster")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.out, args.seed)
