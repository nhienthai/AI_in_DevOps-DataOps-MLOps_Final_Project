# User guide — deploying and operating sentiment-service

Operations manual for the running stack. The `##` sections at the end are alert
runbooks, and their anchors are the `runbook_url` targets in
`prometheus/alerts/*.yml` — Alertmanager links straight here, so the heading names
must not be edited casually.

## Deploying the stack

Requires Docker with Compose v2, roughly 4 GB of free disk for images, and host ports
8000, 5001, 9090, 9093 and 3000 available.

```bash
cp .env.example .env      # then edit the two change-this-before-deployment passwords
docker compose up -d --build
```

The first build downloads CPU-only PyTorch and takes several minutes. Startup order is
enforced by health checks: `api` waits for `mlflow` to be healthy, `prometheus` waits
for `api`.

Verify all six services are healthy:

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

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | |
| Swagger UI | http://localhost:8000/docs | OpenAPI 3.1 at `/openapi.json` |
| MLflow | http://localhost:5001 | postgres-backed, artifacts on a named volume |
| Prometheus | http://localhost:9090 | bound to `127.0.0.1` only |
| Alertmanager | http://localhost:9093 | bound to `127.0.0.1` only |
| Grafana | http://localhost:3000 | `admin` / `GRAFANA_ADMIN_PASSWORD` from `.env` |

Tear down, discarding volumes:

```bash
docker compose down -v
```

Use `-v` whenever you change anything under `prometheus/` or `grafana/provisioning/`.
Grafana provisioning is read at first start and cached in its volume, so a config
change without `-v` appears to do nothing.

## Verifying a deployment

Four checks, in the order that localises a fault fastest.

```bash
# 1. The API is serving and a model is loaded
curl -s localhost:8000/ready
# {"status":"ready","model_version":"2"}

# 2. Prometheus has actually scraped the API
curl -s -G localhost:9090/api/v1/query --data-urlencode 'query=up{job="sentiment-api"}'

# 3. All ten alert rules parsed and loaded
curl -s localhost:9090/api/v1/rules | grep -o '"name":"[A-Za-z]*"' | sort -u

# 4. Both metric families are exposed
curl -s localhost:8000/metrics | grep -c '^http_'
curl -s localhost:8000/metrics | grep -c '^ml_'
```

Check 2 returning an empty `result` array while check 1 succeeds means the API is fine
and the scrape path is broken — look at `prometheus/prometheus.yml` and whether both
containers share a network, not at the API.

## Rolling back to a previous model version

Promotion is by registry stage, so a rollback is a stage transition plus a reload — no
redeploy and no image rebuild.

1. Find the version you want in MLflow at http://localhost:5001, under **Models** →
   `sentiment-service-xlm-roberta`. Note its version number and confirm its metrics.
2. Transition that version to `Production`:

   ```bash
   docker compose exec api python -c "
   import mlflow
   c = mlflow.MlflowClient('http://mlflow:5000')
   c.transition_model_version_stage('sentiment-service-xlm-roberta', '<VERSION>',
                                    'Production', archive_existing_versions=True)
   "
   ```

3. Reload without restarting the container. This requires `SENTIMENT_RELOAD_TOKEN` to
   be set in `.env` — a blank token disables the endpoint entirely.

   ```bash
   curl -X POST localhost:8000/reload -H "x-reload-token: $SENTIMENT_RELOAD_TOKEN"
   ```

4. Confirm the swap:

   ```bash
   curl -s localhost:8000/api/v1/model/info
   ```

A reload warms the replacement model *before* publishing it, so if the rolled-back
version fails to load, the previous model keeps serving and the response reports the
failure. You cannot take the service down with a bad rollback.

If the reload endpoint is disabled, `docker compose restart api` achieves the same
thing with a few seconds of downtime.

## Reading the dashboards

Three dashboards are provisioned from `grafana/dashboards/` into the **Sentiment**
folder, with the datasource provisioned from
`grafana/provisioning/datasources/prometheus.yml`. All of it survives
`docker compose down -v`, because none of it lives in Grafana's volume.

| Dashboard | Read it when | Key panels |
|---|---|---|
| **System & API** | The service is slow or erroring | 5xx share, latency quantiles against the 200 ms SLO, latency heatmap, inference saturation, load shedding |
| **Model & Predictions** | The model is behaving oddly | class distribution and skew, confidence heatmap, drift PSI with the 0.1/0.2 boundaries, input length and truncations, errors by `error_type` |
| **Fairness & Explainability** | Before or after a promotion | max identity-pair delta against the 0.10 gate, gate pass/fail, delta over time by promotion, output balance |

Two habits worth having. On **System & API**, compare `p95 queue wait` against p95
inference on the ML dashboard: high queue with low model time means raise
`max_concurrent_inferences`, the reverse means the model itself is the cost. On
**Fairness**, check the model-version panel first — a delta measured on `stub-*` is an
artefact of the placeholder, not evidence about a trained model.

To query Prometheus directly instead:

| Question | Query |
|---|---|
| Request rate by endpoint | `sum(rate(http_requests_total[5m])) by (endpoint)` |
| p95 latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` |
| Error share | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| Predictions by label | `sum(rate(ml_predictions_total[5m])) by (label)` |
| Confidence distribution | `histogram_quantile(0.5, sum(rate(ml_prediction_confidence_bucket[5m])) by (le))` |
| Low-confidence rate | `rate(ml_low_confidence_total[5m])` |
| Input drift | `ml_drift_psi` |
| Inference saturation | `ml_inference_in_progress`, `ml_inference_queue_depth` |
| Rejected for overload | `rate(ml_inference_overloads_total[5m])` |

---

# Alert runbooks

Each section states what fired, what usually causes it, and the first three commands to
run. Severities come from the rule definitions: `critical` means the service is failing
users now; `warning` means it is degrading; `info` is a reminder.

## APIDown

**Fired:** `up{job="sentiment-api"} == 0` for 1 minute. Severity **critical**.

Prometheus cannot scrape the API at all. This is about reachability, not correctness —
the API may be healthy but unreachable from Prometheus.

**Likely causes:** the `api` container exited or is restarting; it never became healthy
so `depends_on` held it back; the container is up but not listening on 8000; the two
containers are not on the same Compose network.

```bash
docker compose ps api                       # is it up, and is it healthy?
docker compose logs --tail=50 api           # why did it exit or fail its healthcheck?
docker compose exec prometheus wget -qO- http://api:8000/health   # scrape path, from Prometheus
```

If the third command works but the alert persists, the fault is in
`prometheus/prometheus.yml`, not in the API.

## HighErrorRate

**Fired:** 5xx responses exceeded 5% of all requests over 5 minutes, sustained 5
minutes. Severity **critical**.

Note this counts *HTTP* failures. Client mistakes — 422 and 413 — are deliberately
excluded, so this alert never fires because someone sent bad input.

**Likely causes:** the model is not loaded, so every prediction returns 503; inference
is timing out or the pool is saturated and shedding load; an unhandled exception in a
handler; a dependency the handler touches is down.

```bash
curl -s localhost:8000/ready               # 503 here explains everything below
docker compose logs --tail=100 api | grep -iE 'error|traceback|exception'
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=sum(rate(http_requests_total{status=~"5.."}[5m])) by (endpoint,status)'
```

The third command tells you *which* endpoint and status code, which usually names the
cause outright.

## HighLatencyP95

**Fired:** p95 request duration above 500 ms for 5 minutes. Severity **warning**.

The threshold is 2.5× the 200 ms SLO, deliberately loose so that ordinary jitter does
not page anyone.

**Likely causes:** requests queueing behind `max_concurrent_inferences` (default 2);
large batches monopolising the pool; a transformer model on a CPU-only host — with
`xlm-roberta-base` this is the expected cause, and the 200 ms budget is an open risk
rather than a settled property; host CPU contention from other containers.

```bash
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=histogram_quantile(0.95, sum(rate(ml_inference_queue_duration_seconds_bucket[5m])) by (le))'
curl -s -G localhost:9090/api/v1/query --data-urlencode 'query=ml_inference_queue_depth'
docker stats --no-stream sentiment-service-api-1
```

High queue latency with low model latency means raise `max_concurrent_inferences` or
add a replica. Low queue latency with high model latency means the model itself is the
cost — consider the baseline model or a smaller checkpoint.

## ModelNotLoaded

**Fired:** `ml_model_loaded == 0` for 1 minute. Severity **critical**.

The process is alive and serving `/health`, but no predictor is loaded, so every
prediction endpoint returns `503 model_not_ready`. This split is intentional: liveness
stays up so the container is not killed while you diagnose.

**Likely causes:** no model has been promoted to `Production` in the registry; MLflow
is unreachable or holds no promoted version; the checkpoint's `id2label` contradicts
the three-class serving contract and was refused at load time; a reload was attempted
and failed with no previous model to keep.

```bash
curl -s localhost:8000/api/v1/model/info    # reports the load error when there is one
docker compose logs --tail=50 api | grep -iE 'model|load|registry'
docker compose exec api python -c "
import mlflow; c = mlflow.MlflowClient('http://mlflow:5000')
print(c.get_latest_versions('sentiment-service-xlm-roberta', stages=['Production']))"
```

An empty list from the third command means nothing has been promoted, so the registry
backend has nothing to load. Either promote a model with `scripts/validate_model.py`, or
fall back to `SENTIMENT_PREDICTOR_BACKEND=stub` to get the service answering again while
you sort the registry out.

## PredictionSkew

**Fired:** the positive-class share drifted more than 20 percentage points from 0.5,
sustained 15 minutes. Severity **warning**.

The output distribution has moved even though the model has not. That is either a real
change in traffic or a broken model.

**Known limitation:** this rule compares against a 0.5 binary prior, while the model
serves three classes with a heavily under-represented `neutral`. It will need
rewriting per-class when the real model is promoted — tracked with the dataset
reconciliation in W2-04.

**Likely causes:** genuinely shifted traffic (a campaign, a seasonal event, one large
client); an upstream change sending a filtered subset; a mislabelled checkpoint mapping
classes to the wrong ids; a client retry storm amplifying one kind of input.

```bash
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=sum(rate(ml_predictions_total[1h])) by (label)'
curl -s -G localhost:9090/api/v1/query --data-urlencode 'query=ml_drift_psi'
curl -s localhost:8000/api/v1/model/info    # has model_version changed recently?
```

Skew *with* rising PSI points at the inputs. Skew with flat PSI, right after a
deployment, points at the model.

## DriftDetected

**Fired:** `ml_drift_psi > 0.2` for 10 minutes. Severity **warning**.

PSI compares the live input-length distribution against the reference logged beside the
model at training time. 0.2 is the conventional significant-shift boundary — below 0.1
is stable, 0.1–0.2 moderate — which is why the threshold is 0.2 rather than an
arbitrary number.

**Likely causes:** input length genuinely changed (a new client, a new channel, a
different language mix); a truncation or preprocessing change upstream; the model was
promoted without its `drift_reference.json` artifact, so serving fell back to the
bootstrap reference in code and is comparing against the wrong baseline.

```bash
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=histogram_quantile(0.5, sum(rate(ml_input_length_chars_bucket[1h])) by (le))'
curl -s -G localhost:9090/api/v1/query --data-urlencode 'query=rate(ml_input_truncations_total[1h])'
docker compose logs api | grep -i 'drift'    # did it load the artifact or the fallback?
```

Drift is not automatically a fault. Confirm the reference is the right one before
concluding the traffic changed; a fallback reference produces a permanently drifting
signal.

## HighPredictionErrorRate

**Fired:** prediction errors exceeded 5% of predictions over 5 minutes, sustained 5
minutes. Severity **critical**.

Distinct from `HighErrorRate`: this counts failures *inside* the prediction path,
labelled by `error_type`, so it isolates model problems from HTTP problems.

**Likely causes:** inference timeouts (`inference_timeout_seconds`, default 30 s);
overload rejections when the queue budget expires; the model returning a malformed
batch, which `validate_predictions` rejects; tokenisation failing on unusual input.

```bash
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=sum(rate(ml_prediction_errors_total[5m])) by (error_type)'
curl -s -G localhost:9090/api/v1/query --data-urlencode \
  'query=rate(ml_inference_timeouts_total[5m]) or rate(ml_inference_overloads_total[5m])'
docker compose logs --tail=100 api | grep -iE 'timeout|overload|validate_predictions'
```

The `error_type` label from the first command routes the rest of the investigation:
`model_not_ready` → see [ModelNotLoaded](#modelnotloaded); `inference_timeout` or
`inference_overloaded` → see [HighLatencyP95](#highlatencyp95).

## ModelStale

**Fired:** the model has not been reloaded in over 7 days (604800 s), sustained 1 hour.
Severity **info**.

A reminder, not an incident. Nothing is broken; the question is whether the model
should still be the one serving.

**Likely causes:** no retraining has run; retraining ran but promotion did not; a
promotion happened but the API was never reloaded, so it is still holding the old
model in memory.

```bash
curl -s localhost:8000/api/v1/model/info    # run_id and trained_at of what is loaded
docker compose exec api python -c "
import mlflow; c = mlflow.MlflowClient('http://mlflow:5000')
print([(v.version, v.current_stage) for v in
       c.search_model_versions(\"name='sentiment-service-xlm-roberta'\")])"
curl -s -G localhost:9090/api/v1/query --data-urlencode 'query=ml_model_last_reload_timestamp'
```

If a newer `Production` version exists that the API is not serving, reload it — see
[Rolling back to a previous model version](#rolling-back-to-a-previous-model-version),
which is the same procedure in the other direction.

## FairnessRegression

**Fired:** `ml_fairness_max_delta > 0.1` for 5 minutes. Severity **critical**.

The deployed model's worst identity-pair gap exceeds the same 0.10 threshold that
`scripts/validate_model.py` enforces at promotion time. Something got past the gate, or
the gate was bypassed.

**Likely causes:** a model was promoted by hand rather than through
`validate_model.py`; the promoted run carried a fairness metric measured on a different
probe set; a mitigation was removed — most likely identity blinding was switched off,
since that is what holds the delta at exactly zero.

```bash
curl -s localhost:8000/api/v1/model/info                  # which model, and its fairness_delta
python scripts/run_fairness_probe.py --threshold 0.1      # re-measure the live service
docker compose exec api python -c "
import mlflow; c = mlflow.MlflowClient('http://mlflow:5000')
print([(v.version, v.current_stage) for v in
       c.search_model_versions(\"name='sentiment-service-xlm-roberta'\")])"
```

The probe exits 1 when the gate fails and prints the worst pairs, which names the axis
to investigate. Roll back to the previous `Production` version while you do — see
[Rolling back to a previous model version](#rolling-back-to-a-previous-model-version).
The measured before/after figures for each mitigation are in
[`docs/FAIRNESS.md`](FAIRNESS.md).

## FairnessUnmeasured

**Fired:** a model is loaded and serving traffic, but `ml_fairness_max_delta` is still
zero after 30 minutes. Severity **info**.

Not necessarily a fault — it is genuinely ambiguous, which is why it is `info` rather
than a page. Zero means either "no measurement has been attached" or "measured, and
actually zero". Identity blinding drives the delta to exactly zero by construction, so
a blinded model sits here legitimately and permanently.

**Likely causes:** the promoted MLflow run has no `fairness_max_delta` metric, so the
serving layer had nothing to publish; the probe has never been run against this model;
or identity blinding is active and the zero is real.

```bash
curl -s localhost:8000/api/v1/model/info                  # fairness_delta null vs 0.0
python scripts/run_fairness_probe.py --log-to-mlflow      # measure and record it
docker compose logs api | grep -i fairness
```

`fairness_delta: null` in `/model/info` distinguishes the two cases: null means
unmeasured, `0.0` means measured as zero. If it is null, run the probe with
`--log-to-mlflow --run-id <the run behind the deployed model>` so the number travels
with the model rather than living only in a terminal.

## See also

- [`README.md`](../README.md) — quickstart and troubleshooting
- [`docs/FAIRNESS.md`](FAIRNESS.md) — fairness before/after, with the accuracy cost
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — why the system is shaped this way
- [`docs/api.md`](api.md) — endpoint reference
- [`docs/TESTING_STRATEGY.md`](TESTING_STRATEGY.md) — the four test types
