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

Accepts `{"text": "...", "method": "lime"}` and returns token attributions for the model
currently serving. The explainer is built on demand from the active predictor and rebuilt
whenever a reload replaces it, so an explanation never describes a model that is no longer
answering `/predict`.

The response carries the predicted `label`, the positive-class `score`, the
`model_version`, and an `attributions` list of `{token, attribution}` — signed, so a
negative value means the token pushed the prediction away from positive.

Returns `501 explainer_not_available` only when no explainer can be built: a deployment
without `lime` installed, or one with no model loaded.

Note the scope of the explanation: because the `Predictor` protocol exposes a single
positive-class `score` rather than the full three-class distribution, LIME explains *the
positive-class probability* rather than attributing across all three classes.

Global SHAP feature importance is not served here — it describes the model rather than a
request, and is logged as an MLflow artifact on the training run.

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
  promoted artifact and attaches its MLflow metrics and provenance. It dispatches on what
  the run logged — a `*.joblib` file loads the TF-IDF baseline, anything else is treated as
  a Hugging Face directory. Production deployment sets
  `SENTIMENT_PREDICTOR_BACKEND=registry`.
- M2 local loading: `sentiment.models.local.load_local_predictor` serves a Hugging Face
  directory straight from `SENTIMENT_LOCAL_MODEL_DIR`, reading version, metrics and
  provenance from the `serving_metadata.json` written next to the weights. This is how both
  fine-tuned transformers ship, since their weights never entered MLflow.
- M2 input format: `serving_metadata.json` may carry a `preprocessing` block
  (`clean_dataset_artifacts`, `template`) that `sentiment.models.text_format.InputFormat`
  applies before tokenizing. A model trained on shaped input — PhoBERT-v2 is trained on
  `"Chủ đề: {topic} | {cleaned}"` — must be served the same shape, so the transformation
  travels with the weights rather than living in a training notebook.
- M1 drift artifact: place `drift_reference.json` alongside the registered model artifact.
  Its length bins, frequencies, and positive prior are loaded with the predictor; a safe
  bootstrap reference remains available for older artifacts.
- M5 explanation: an explainer is built automatically from the active predictor. Passing
  `explainer=` or `explainer_factory=` to `create_app` overrides it, which is what the
  tests do.
- Fairness: the promoted model's `fairness_max_delta` metric is published as the
  `ml_fairness_max_delta` gauge at load time, which is what `FairnessRegression` alerts on.
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
