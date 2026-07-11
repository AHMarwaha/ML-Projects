"""API tests: train a tiny model, then exercise /health, /predict, /drift."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("artifacts")
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"), "--out", str(artifacts)],
        check=True,
    )
    os.environ["ARTIFACT_DIR"] = str(artifacts)
    from serve import app  # imported after ARTIFACT_DIR is set

    with TestClient(app) as c:
        yield c


def _record(**overrides):
    base = {
        "hour_sin": 0.5, "hour_cos": 0.87, "dow_sin": 0.0, "dow_cos": 1.0,
        "doy_sin": 0.2, "doy_cos": 0.98, "is_weekend": 0, "temperature_c": 8.0,
        "lag_1h": 2100.0, "lag_2h": 2080.0, "lag_3h": 2060.0,
        "lag_24h": 2050.0, "lag_168h": 2000.0, "rolling_24h_mean": 2070.0,
    }
    base.update(overrides)
    return base


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_predict_returns_plausible_value(client):
    r = client.post("/predict", json={"records": [_record()]})
    assert r.status_code == 200
    pred = r.json()["predictions_mw"][0]
    assert 500 < pred < 5000  # sane megawatt range for the synthetic system


def test_predict_rejects_malformed(client):
    r = client.post("/predict", json={"records": [{"hour_sin": "not a number"}]})
    assert r.status_code == 422


def test_drift_needs_enough_records(client):
    r = client.post("/drift", json={"records": [_record()] * 5})
    assert r.status_code == 422
