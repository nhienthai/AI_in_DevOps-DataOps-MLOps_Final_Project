# sentiment-service

Vietnamese sentiment analysis as a production ML system: a fine-tuned XLM-RoBERTa
classifier served over a versioned REST API, tracked in MLflow, instrumented for
Prometheus, and deployed as six Docker Compose services.

[![CI Pipeline](https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Coverage gate](https://img.shields.io/badge/coverage-91%25%20%2F%2085%25%20gate-brightgreen)

DDM501 — AI in Production: From Models to Systems. Final project, Topic 8.

> **Status: serving the fine-tuned transformer.** Six healthy services running
> PhoBERT-v2 on UIT-VSFC — held-out **macro-F1 0.8457 / accuracy 0.9416**, ahead of
> the XLM-RoBERTa it replaced (0.8337 / 0.9359), which stays mounted and one
> environment variable away. Both were fine-tuned on a Kaggle GPU and are served
> from disk with `SENTIMENT_PREDICTOR_BACKEND=local`; see
> [Serving the fine-tuned transformers](#serving-the-fine-tuned-transformers). The TF-IDF
> baseline it replaced (**macro-F1 0.7149 / accuracy 0.8629**, worst identity-pair gap
> **0.0000** after mitigation) went through the full promotion gate — macro-F1,
> accuracy, latency **and** fairness — and remains registry version 2. See
> [Current state](ARCHITECTURE.md#current-state) for what is and is not built.

## Architecture at a glance

```
   client ──HTTP──▶ ┌──────────────────────────────┐ ◀── MLflow registry (Production)
                    │  api  (FastAPI + uvicorn)    │      backed by postgres
                    │   /api/v1/predict            │
                    │   /api/v1/predict/batch      │
                    │   /api/v1/explain            │
                    │   /api/v1/model/info         │
                    │   /health  /ready  /metrics  │
                    └──────────────┬───────────────┘
                                   │ scrape /metrics
                          ┌────────▼───────┐      ┌──────────┐
                          │   prometheus   │─────▶│ grafana  │
                          └───────┬────────┘      └──────────┘
                                  │ 10 alert rules
                          ┌───────▼────────┐
                          │  alertmanager  │
                          └────────────────┘
```

Full diagrams, the component table and every technology trade-off are in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Quickstart

Requires Docker with Compose v2 and host ports 8000, 5001, 9090, 9093, 3000 free.

```bash
cp .env.example .env      # then set the two change-this-before-deployment passwords
docker compose up -d --build
```

The first build downloads CPU-only PyTorch, so expect several minutes. Watch the
services become healthy:

```bash
docker compose ps --format "table {{.Service}}\t{{.Status}}"
```

```
SERVICE        STATUS
alertmanager   Up 7 minutes (healthy)
api            Up 7 minutes (healthy)
grafana        Up 10 seconds (healthy)
mlflow         Up 6 minutes (healthy)
postgres       Up 7 minutes (healthy)
prometheus     Up 20 seconds (healthy)
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana | http://localhost:3000 (`admin` / `GRAFANA_ADMIN_PASSWORD`) |

Tear down with `docker compose down -v`.

## API examples

Every response below is real output, captured from the stack above serving the
promoted baseline (registry version 2).

### Single prediction

```bash
curl -X POST localhost:8000/api/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"Giáo viên dạy rất hay và nhiệt tình."}'
```

```json
{
  "label": "positive",
  "score": 0.9990086381724512,
  "confidence": 0.9990086381724512,
  "model_version": "2",
  "truncated": false,
  "latency_ms": 9.286
}
```

A negative example, showing that `score` is the positive-class probability while
`confidence` belongs to the predicted class:

```bash
curl -X POST localhost:8000/api/v1/predict \
  -H 'content-type: application/json' \
  -d '{"text":"Bài giảng nhàm chán, tài liệu sơ sài."}'
```

```json
{
  "label": "negative",
  "score": 0.05865991724982634,
  "confidence": 0.7965254433567068,
  "model_version": "2",
  "truncated": false,
  "latency_ms": 1.886
}
```

### Batch prediction

Invalid items do not fail the batch. Each result carries exactly one of `prediction` or
`error`, and results stay in request order:

```bash
curl -X POST localhost:8000/api/v1/predict/batch \
  -H 'content-type: application/json' \
  -d '{"texts":["Bài giảng rất dễ hiểu.","","Phòng học hơi nóng."]}'
```

```json
{
  "results": [
    {
      "index": 0,
      "prediction": {
        "label": "positive",
        "score": 0.9834527816532195,
        "confidence": 0.9834527816532195,
        "model_version": "2",
        "truncated": false,
        "latency_ms": 8.012
      },
      "error": null
    },
    { "index": 1, "prediction": null, "error": "Text must not be blank." },
    {
      "index": 2,
      "prediction": {
        "label": "negative",
        "score": 0.022738896515311413,
        "confidence": 0.7562829493932797,
        "model_version": "2",
        "truncated": false,
        "latency_ms": 8.012
      },
      "error": null
    }
  ]
}
```

### Explaining a prediction

```bash
curl -X POST localhost:8000/api/v1/explain \
  -H 'content-type: application/json' \
  -d '{"text":"Giáo viên dạy rất hay nhưng tài liệu thì sơ sài."}'
```

```json
{
  "method": "lime",
  "label": "negative",
  "score": 0.42249894047508385,
  "model_version": "2",
  "attributions": [
    { "token": "rất", "attribution": 0.30225089719036674 },
    { "token": "hay", "attribution": 0.25988701246365237 },
    { "token": "nhưng", "attribution": -0.19009493526199378 },
    { "token": "thì", "attribution": -0.16317639194310055 }
  ]
}
```

The model reads the concession correctly: `rất` and `hay` push toward positive, `nhưng`
pulls the other way, and the sentence lands negative. Global SHAP importance for the
same model is logged as an MLflow artifact on its training run.

### Model info

```bash
curl localhost:8000/api/v1/model/info
```

```json
{
  "model_version": "2",
  "stage": "Production",
  "predictor_class": "BaselinePredictor",
  "metrics": {
    "test_macro_f1": 0.7148894704416761,
    "test_accuracy": 0.8629185091598232,
    "cv_macro_f1_mean": 0.7138161812639017,
    "cv_macro_f1_std": 0.004312037100421051,
    "fairness_max_delta": 0.0,
    "data_train_rows": 11426.0,
    "data_train_min_class_share": 0.040084018904253456
  },
  "fairness_delta": 0.0,
  "trained_at": null,
  "run_id": "e9164e638bfc4417b2ba76a23f85d259",
  "build_revision": "development"
}
```

`metrics` is abridged here — the live response carries all 30, including per-dimension
fairness deltas and the dataset fingerprint. `run_id` resolves in MLflow to the params,
artifacts and data that produced this exact model.

### Health and readiness

```bash
curl localhost:8000/health   # {"status":"ok"}
curl localhost:8000/ready    # {"status":"ready","model_version":"2"}
```

`/health` reports process liveness and stays 200 even when model loading failed;
`/ready` returns `503 model_not_ready` until a predictor is warm. Keeping them separate
means a model problem does not get the container killed while you diagnose it.

Full endpoint reference: [`docs/api.md`](docs/api.md).

## Dashboards and alerts

Three dashboards provision themselves from `grafana/dashboards/` into the **Sentiment**
folder — System & API, Model & Predictions, and Fairness & Explainability. Ten alert
rules load from `prometheus/alerts/`, each linking to a runbook in
[`docs/user-guide.md`](docs/user-guide.md).

To give the panels something to show:

```bash
python scripts/load_test.py --scenario steady --duration 120 --rps 20
python scripts/load_test.py --scenario drift --duration 120   # fires DriftDetected
```

## Training and promotion

The baseline trains on CPU in under a minute. Nothing is promoted by hand — the gate
checks macro-F1, accuracy, latency and fairness, and refuses on any of them.

```bash
docker compose --profile training up trainer     # or, locally:

python scripts/train_model.py --model-type baseline --mitigation blinding \
  --tune-trials 10 --cv-splits 5 --explain

python scripts/validate_model.py \
  --model-path ./artifacts/baseline_model.joblib --model-type baseline \
  --min-macro-f1 0.70 --min-accuracy 0.85 --max-fairness-delta 0.10 \
  --run-id <RUN_ID>
```

Then point the API at the registry instead of the stub:

```bash
SENTIMENT_PREDICTOR_BACKEND=registry docker compose up -d api
```

Measure the deployed model's fairness over HTTP at any time:

```bash
python scripts/run_fairness_probe.py --threshold 0.10   # exits 1 if the gate fails
```

## Serving the fine-tuned transformers

Both transformers were fine-tuned on Kaggle, not in this stack, and arrive as
archives in `models/` that git ignores. Unpack them, then log the runs into MLflow:

```bash
python scripts/setup_local_model.py          # both bundles -> artifacts/
python scripts/import_donated_run.py         # XLM-R run from its tracking database
python scripts/import_donated_run.py --from-model-metadata \
  --model-dir artifacts/phobert-sota         # PhoBERT, which arrived without one
```

| Model | Directory | accuracy | macro-F1 |
|---|---|---:|---:|
| **PhoBERT-v2** (served) | `artifacts/phobert-sota` (0.50 GiB) | **0.9416** | **0.8457** |
| XLM-RoBERTa | `artifacts/xlm-roberta` (1.05 GiB) | 0.9359 | 0.8337 |

Measured on the 3166-example UIT-VSFC test split, through the same input format
the API serves. `setup_local_model.py` takes only the model root of each archive:
the two bundles also carry training checkpoints holding `optimizer.pt` state —
8.7 GB combined — which matter only for resuming a fine-tune.

Serve, and switch between them with one variable:

```bash
SENTIMENT_PREDICTOR_BACKEND=local docker compose up -d --build api
SENTIMENT_LOCAL_MODEL_DIR=/app/artifacts/xlm-roberta docker compose up -d api   # the other one
```

```console
$ curl -s localhost:8000/api/v1/model/info | jq '{model_version, stage, test_macro_f1: .metrics.test_macro_f1}'
{
  "model_version": "local-phobert-sota",
  "stage": "Production",
  "test_macro_f1": 0.8457
}
```

Three things are worth knowing about this path:

- **PhoBERT is not served the raw request text.** Its notebook trains on
  `"Chủ đề: {topic} | {cleaned sentence}"`, so a bare sentence is out of
  distribution for it — macro-F1 drops from 0.8457 to 0.7956, below the model it
  replaced. The HTTP contract has no topic field, so serving pins the dataset's
  own `others` default, which measured marginally *better* than passing the true
  topic. The transformation is declared in `serving_metadata.json` next to the
  weights and applied by `sentiment.models.text_format.InputFormat`, so it cannot
  drift away from the weights it belongs to.
- **Weights are not uploaded to MLflow.** `import_donated_run.py` logs the run and
  its metrics, then registers a version whose artifact is a JSON pointer at the
  model directory. That gives the registry accurate lineage without pushing a
  gigabyte through it on every deploy — but it also means
  `SENTIMENT_PREDICTOR_BACKEND=registry` cannot serve these models. Use `local`.
- **The donated run measured no fairness metric,** so `/model/info` reports
  `fairness_delta: null` and the `FairnessUnmeasured` alert fires by design. Run
  `scripts/run_fairness_probe.py` against the deployed model to get that number.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
make install      # editable install + dev dependencies
make lint         # flake8 + black --check + isort --check-only
make typecheck    # mypy
make test         # pytest with the coverage gate
make smoke        # integration tests against a running stack
pytest -m slow    # model behaviour tests; downloads the dataset
```

Configuration comes from `SENTIMENT_`-prefixed environment variables or `.env`. See
[`src/sentiment/config.py`](src/sentiment/config.py) for every setting and its default.

## Troubleshooting

**Port 8000 already in use.** The API cannot bind and the container exits immediately.

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN     # find the holder
```

Either stop it, or change the host side of the mapping in `docker-compose.yml`
(`"8001:8000"`). The same applies to 9090, 9093, 3000 and 5001 — a second monitoring
stack on the same machine is the usual culprit, and the failure message names the port.
Note that `docker compose up | tail` hides the real exit code, so a port conflict can
look like a successful start; read the output rather than trusting the exit status.

**MLflow is published on 5001, not 5000.** macOS AirPlay Receiver binds `*:5000` and
wins over Docker's loopback publish, so every request is answered by AirTunes and a
healthy MLflow returns 403. Rather than asking each developer to turn AirPlay off, the
stack publishes `127.0.0.1:5001:5000`; the server still listens on 5000 inside the
network, which is why `SENTIMENT_MLFLOW_TRACKING_URI` stays `http://mlflow:5000` for the
services and `http://localhost:5001` for anything run from the host.

If a URL ever answers 403 with `Server: AirTunes`, that is this conflict — you are on
5000 by mistake:

```bash
curl -sD - -o /dev/null http://localhost:5001/    # 200 from MLflow
docker compose exec api python -c \
  "import urllib.request; print(urllib.request.urlopen('http://mlflow:5000/health').status)"
```

**`mlflow` unhealthy because postgres was slow to start.** `mlflow` depends on postgres
being healthy, and `api` depends on `mlflow`. On a cold machine postgres can exceed its
start period, which cascades.

```bash
docker compose logs postgres | tail -20
docker compose restart mlflow api
```

Restarting the two dependents is enough; a full `down -v` throws away the MLflow
database for no reason.

**Provisioning changes appear to do nothing.** Grafana reads
`grafana/provisioning/` at first start and then caches it in its volume, so editing a
datasource or dashboard file has no visible effect on a restart.

```bash
docker compose down -v && docker compose up -d
```

The `-v` is the whole point — without it the old provisioning survives.

**Dataset download times out.** `tridm/UIT-VSFC` is fetched from the HuggingFace hub at
training time, and a slow or blocked connection stalls the run rather than failing it
fast. Tests never download anything: they use fixtures, so CI is unaffected. For
training, pre-fetch once and let the cache serve subsequent runs:

```bash
python -c "from datasets import load_dataset; load_dataset('tridm/UIT-VSFC')"
```

The Kaggle notebook in [`notebooks/`](notebooks/) avoids this entirely by training on
Kaggle's GPU with its own network.

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROBLEM.md`](docs/PROBLEM.md) | Problem statement, users, requirements, success metrics with targets |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, data flow, technology trade-offs, what is not built yet |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Every run, what it showed, and the mistakes worth recording |
| [`docs/FAIRNESS.md`](docs/FAIRNESS.md) | Measured bias, two mitigations, and what each one cost |
| [`docs/ETHICS.md`](docs/ETHICS.md) | Privacy, failure modes, and uses this system refuses |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Team roles, what each member built, branching, PR checklist |
| [`docs/deck.html`](docs/deck.html) | The defence deck — open in a browser, arrow keys to present |
| [`docs/PRESENTATION.md`](docs/PRESENTATION.md) | Slide-by-slide plan, speaker notes and the demo script |
| [`docs/REHEARSAL.md`](docs/REHEARSAL.md) | Clean-machine checklist and what a dry run already found |
| [`docs/QA_PREP.md`](docs/QA_PREP.md) | Three questions and answers per member |
| [`docs/api.md`](docs/api.md) | Endpoint reference |
| [`docs/user-guide.md`](docs/user-guide.md) | Deployment, rollback, and a runbook per alert |
| [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md) | The four test types and what each catches |
| [`docs/KAGGLE_GUIDE.md`](docs/KAGGLE_GUIDE.md) | Fine-tuning on Kaggle's free GPU |
| [`TASKS.md`](TASKS.md) | Task board, owners, current status |

## Future work

Deliberately out of scope for this timeline, with the reasoning recorded in
[`ARCHITECTURE.md`](ARCHITECTURE.md) rather than discovered later:

- **Feedback loop and continuous retraining** — a `POST /api/v1/feedback` endpoint plus
  a scheduler. It roughly doubles the service count for a narrative benefit the drift
  metric already provides.
- **Horizontal scaling** — the API is stateless apart from its in-memory drift window,
  which would need a shared store.
- **Aspect-based sentiment** — needs a differently annotated dataset.
- **Sarcasm detection** — discussed as a known failure mode rather than solved.
