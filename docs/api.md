# Sentiment Service API

The running service publishes the authoritative OpenAPI 3.1 document at
`GET /openapi.json` and interactive Swagger documentation at `GET /docs`.

Base URL for local development: `http://localhost:8000`.

## Operational endpoints

### `GET /health`

Process liveness. It returns `200 {"status":"ok"}` even if model loading failed.

### `GET /ready`

Serving readiness. It returns `200` only after a predictor is loaded; otherwise it returns
a typed `503 model_not_ready` error.

### `GET /metrics`

Prometheus text exposition for `http_*` and `ml_*` metric families.

## Inference endpoints

### `POST /api/v1/predict`

Request:

```json
{"text": "Arrived quickly and works perfectly."}
```

Response fields:

- `label`: `positive`, `neutral`, or `negative`.
- `score`: probability of the positive class.
- `confidence`: `max(score, 1 - score)`.
- `model_version`: traceable serving-model version.
- `truncated`: whether the model shortened the input.
- `latency_ms`: server-side inference time.

Blank text returns `422`; input above `SENTIMENT_MAX_TEXT_LENGTH` returns `413`.

### `POST /api/v1/predict/batch`

Accepts `{"texts": [...]}` with one to 64 items by default. A malformed item produces an
error only in that result position; other items are still classified. A batch above the
configured limit returns `413`.

### `GET /api/v1/model/info`

Reports the serving version, lifecycle stage, predictor class, evaluation metrics,
fairness delta, training timestamp, MLflow run ID, and application build revision.

### `POST /reload`

When `SENTIMENT_RELOAD_TOKEN` is configured, send it in `X-Reload-Token` to load and warm
a replacement registry model. The active model remains available during loading and is
replaced atomically only after validation succeeds. A failed reload leaves the old model
active. This operations endpoint should not be exposed outside the trusted control plane.

### `POST /api/v1/explain`

Accepts `{"text": "...", "method": "lime"}`. M3 owns this HTTP contract; M5 supplies the
LIME implementation through `sentiment.serving.predictor.Explainer`. Until an explainer is
installed, the endpoint returns typed status `501 explainer_not_available`.

## Error contract

Every expected failure uses the same body:

```json
{
  "error_code": "text_too_long",
  "message": "Text exceeds 5000 characters.",
  "request_id": "a-correlation-id"
}
```

The correlation ID is also returned in the `X-Request-ID` response header. Raw review text
is not used as a metric label or included in API logs.

## Integration contracts for other owners

- M2 registry loading: `sentiment.models.registry.load_production_predictor` downloads the
  promoted Hugging Face artifact and attaches its MLflow metrics and provenance. Production
  deployment sets `SENTIMENT_PREDICTOR_BACKEND=registry`.
- M1 drift artifact: place `drift_reference.json` alongside the registered model artifact.
  Its length bins, frequencies, and positive prior are loaded with the predictor; a safe
  bootstrap reference remains available for older artifacts.
- M5 explanation: inject an object satisfying `sentiment.serving.predictor.Explainer` when
  constructing the app.
- M4 monitoring: provide the configuration mounted from `prometheus/`, `alertmanager/`, and
  `grafana/` by `docker-compose.yml`.

## Runtime behaviour

- Labels are `negative`, `neutral`, or `positive`.
- `score` is the positive-class probability; `confidence` is the predicted-class probability.
- Valid batch items are passed to the model in one call and restored to request order.
- Synchronous model execution runs in a bounded worker pool rather than on the FastAPI event
  loop. Queue saturation returns `429`; execution timeout returns `504`.
- Readiness is published only after a warm-up prediction validates the model contract.
- The serving image installs `requirements-serving.txt`; training-only dependencies stay out
  of the runtime image.
