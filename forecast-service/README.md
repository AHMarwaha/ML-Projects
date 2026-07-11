# Forecasting as a Service: Production ML Pipeline

An end-to-end machine learning pipeline that takes a demand-forecasting model
from raw data to a monitored, containerised REST service. Built to
demonstrate the full model lifecycle: training with leakage-safe evaluation,
experiment tracking, serving with input validation, automated testing in CI,
and statistical drift detection.

## Architecture

```
data.py ──> features.py ──> train.py ──────────┐
 (hourly      (cyclical       (walk-forward     │ model.joblib
  demand       encodings,      evaluation,      │ drift_reference
  series)      leak-safe       MLflow logging)  │
               lags)                            v
                                            serve.py (FastAPI)
                                            ├── GET  /health
                                            ├── POST /predict
                                            └── POST /drift ──> drift.py
                                                                (PSI + KS test)
```

## Key design decisions

**Walk-forward evaluation, not random splits.** Random cross-validation on
time series leaks future information into training and overstates accuracy.
`train.py` evaluates on four expanding-window folds, each tested on the four
weeks immediately following its training data, which is exactly how the model
is used once deployed.

**Leakage-safe features.** Every feature is computable at prediction time:
cyclical time encodings, temperature, and lags built strictly from past
values (`shift(1)` before any rolling aggregate). A unit test asserts the
lag alignment.

**Drift detection as a first-class endpoint.** `/drift` compares incoming
feature distributions against a stored training reference using the
Population Stability Index and the two-sample Kolmogorov-Smirnov test per
feature. PSI > 0.25 or KS p < 0.01 flags a feature; any flagged feature sets
`drift_detected`, the signal that would gate a retraining job.

**Optional MLflow.** Training logs parameters, per-fold metrics, and the
model artifact to MLflow when it is installed, and degrades gracefully when
it is not, so tests and CI stay lightweight.

**Validated inputs.** Pydantic schemas reject malformed requests with a
field-level error before anything reaches the model.

## Quickstart

```bash
pip install -r requirements-dev.txt

# Train (walk-forward evaluation + artifact export; logs to MLflow if present)
python src/train.py --out artifacts

# Serve
uvicorn serve:app --app-dir src --reload

# Predict
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "records": [{
    "hour_sin": 0.5, "hour_cos": 0.87, "dow_sin": 0.0, "dow_cos": 1.0,
    "doy_sin": 0.2, "doy_cos": 0.98, "is_weekend": 0, "temperature_c": 8.0,
    "lag_1h": 2100, "lag_2h": 2080, "lag_3h": 2060,
    "lag_24h": 2050, "lag_168h": 2000, "rolling_24h_mean": 2070
  }]
}'
```

### Docker

```bash
docker build -t forecast-service .
docker run -p 8000:8000 forecast-service
curl localhost:8000/health
```

The image trains at build time so it is fully self-contained. In a real
deployment the model would be pulled from a registry (MLflow Model Registry
or object storage) at startup instead; the `ARTIFACT_DIR` environment
variable is the seam for that change.

### Tests

```bash
pytest tests/ -v
```

Covers feature engineering (shape, NaNs, lag alignment, encoding bounds),
drift detection (no false alarm on identical distributions, detection of a
mean shift), and the API end to end (train a model, then exercise /health,
/predict, and /drift including malformed-input rejection).

CI (GitHub Actions) runs the test suite, builds the Docker image, and smoke
tests the running container on every push.

## Data

The repository ships with a synthetic hourly electricity-demand generator
(`data.py`) with daily/weekly/annual seasonality, temperature dependence, a
trend, and heteroscedastic noise, keeping the project reproducible with zero
credentials. The pipeline is data-source agnostic: to use real demand data,
replace `generate_demand()` with a loader for e.g. the ENTSO-E transparency
platform and keep the schema (`demand_mw`, `temperature_c`, hourly index).

## Repository layout

```
src/
  data.py       synthetic demand data generation
  features.py   leakage-safe feature engineering
  train.py      walk-forward training + MLflow tracking
  drift.py      PSI + Kolmogorov-Smirnov drift detection
  serve.py      FastAPI service (health, predict, drift)
tests/          unit + API tests (pytest)
.github/        CI workflow: tests, Docker build, container smoke test
Dockerfile
```

Python 3.10+.
