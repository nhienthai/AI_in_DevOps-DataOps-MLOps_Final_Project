# sentiment-service — Design Specification

**Course:** DDM501 — AI in Production: From Models to Systems
**Deliverable:** Final Project (40% of course grade)
**Date:** 2026-08-09
**Team size:** 5 (instructor-approved exception to the 3–4 stated in the brief)
**Topic:** #8 — Sentiment Analysis Service

---

## 1. Problem Definition & Requirements

### 1.1 Business context

An e-commerce retailer receives more product reviews per day than its customer-experience
(CX) team can read. Negative reviews left unanswered for more than 48 hours correlate with
customer churn and with public escalation on social channels. Today, triage is manual and
first-in-first-out, so a scathing review can sit behind two hundred neutral ones.

`sentiment-service` scores every incoming review in real time and returns a sentiment label
with a confidence score, so the CX queue can be ordered by *urgency* rather than arrival
time.

### 1.2 Problem statement

> Given the free text of a product review, classify its sentiment as positive or negative in
> under 200 ms at p95, with sufficient accuracy and demonstrated demographic fairness that
> the output can be used to prioritise human CX work without introducing systematic bias
> against any group of reviewers.

### 1.3 Users and use cases

| User | Use case |
|---|---|
| CX agent | Sees a queue ordered by negative-sentiment confidence, works the worst first. |
| CX manager | Watches a dashboard of sentiment volume and skew to staff shifts. |
| Data scientist | Inspects token-level explanations to understand and challenge a prediction. |
| Platform engineer | Monitors latency, error rate, and input drift; receives alerts. |

### 1.4 Requirements

**Functional**

| ID | Priority | Requirement |
|---|---|---|
| F1 | Must | Classify a single review text as positive/negative with a confidence score. |
| F2 | Must | Batch endpoint accepting up to 64 texts, tolerant of per-item failure. |
| F3 | Must | Expose model provenance (version, metrics, training date) over HTTP. |
| F4 | Must | Return a token-level explanation for any given text on demand. |
| F5 | Must | Expose Prometheus metrics covering both system and model behaviour. |
| F6 | Should | Separate liveness and readiness signals for orchestration. |
| F7 | Could | Accept end-user feedback on a prediction (deferred; see §9). |

**Non-functional**

| ID | Priority | Requirement |
|---|---|---|
| N1 | Must | p95 end-to-end latency < 200 ms for single prediction on CPU. |
| N2 | Must | Sustain ≥ 50 req/s on a single API replica. |
| N3 | Must | Entire stack starts from a clean clone with one command. |
| N4 | Must | No review text written to production logs (hashes and lengths only). |
| N5 | Must | Container runs as a non-root user. |
| N6 | Should | Cold start (process up → `/ready` returns 200) under 30 s. |

### 1.5 Success metrics

**Business**

- Median time-to-first-response on negative reviews reduced (baseline measured pre-launch).
- ≥ 90% of reviews the model labels negative with confidence > 0.9 are triaged within 1 hour.

**System**

- p95 latency < 200 ms; p99 < 500 ms.
- Availability ≥ 99% measured over the demo window.
- Error rate (5xx) < 1%.

**Model**

- Macro-F1 ≥ 0.92 on the held-out Amazon Polarity test split.
- ROC-AUC ≥ 0.97.
- **Fairness: max |Δ mean sentiment score| across Equity Evaluation Corpus identity pairs
  ≤ 0.05.** This is a promotion gate, not a report line — see §4.4.

### 1.6 Scope and constraints

**In scope:** English-language binary sentiment; batch and single online inference; SHAP and
LIME explanations; EEC fairness auditing; full MLOps stack (tracking, serving, monitoring,
CI/CD).

**Out of scope (explicitly, with rationale):**

- *Multi-language.* The rubric lists it as a challenge, not a requirement; it would double
  the data and evaluation work for no additional rubric coverage. Documented as future work.
- *Aspect-based sentiment.* Requires a differently-annotated dataset.
- *Sarcasm detection as a distinct capability.* Discussed as a known failure mode in the
  ethics section rather than solved.
- *Kubernetes.* Docker Compose is what the brief requires.
- *Feedback-driven continuous retraining.* Deferred (§9); adds a scheduler and a queue for a
  narrative benefit that the drift metric already provides.

**Constraints:** 4 calendar weeks; CPU-only inference assumed (no GPU in CI or on demo
hardware); GitHub Actions free-tier runner limits mean transformer training cannot run on
every push.

---

## 2. System Design & Architecture

### 2.1 High-level architecture

```
        ┌───────────── TRAINING (docker compose --profile training, one-shot) ─────────────┐
        │                                                                                  │
        │  HF amazon_polarity                                                              │
        │        │                                                                         │
        │        ▼                                                                         │
        │    ingest ──▶ raw parquet ──▶ validate ──▶ preprocess ──▶ train ─┬─ baseline     │
        │                                  │            │                  └─ distilbert   │
        │                          quality gate    stratified                    │         │
        │                          (fails run)     subsample, seeded             │         │
        │                                                                        ▼         │
        │                                                          evaluate + fairness     │
        │                                                                        │         │
        │                                                                        ▼         │
        │                                          MLflow tracking ──────▶ postgres        │
        │                                          artifacts ───────────▶ named volume     │
        │                                                    │                             │
        │                                    registry: promote → Staging → Production      │
        └────────────────────────────────────────────────────┼─────────────────────────────┘
                                                             │ loaded at API startup
   client ──HTTP──▶ ┌──────────────────────────────┐ ◀───────┘
                    │  api  (FastAPI, uvicorn)     │
                    │   /api/v1/predict            │
                    │   /api/v1/predict/batch      │
                    │   /api/v1/explain            │
                    │   /api/v1/model/info         │
                    │   /health  /ready  /metrics  │
                    │                              │
                    │  in-process instrumentation: │
                    │   latency · class dist ·     │
                    │   confidence · PSI drift     │
                    └──────────────┬───────────────┘
                                   │ scrape /metrics
                                   ▼
                          ┌────────────────┐      ┌──────────┐
                          │   prometheus   │─────▶│ grafana  │
                          └───────┬────────┘      └──────────┘
                                  │ alert rules
                                  ▼
                          ┌────────────────┐
                          │  alertmanager  │
                          └────────────────┘
```

### 2.2 Components and responsibilities

| Component | Responsibility | Depends on |
|---|---|---|
| `data/` | Download, validate, and split raw data. Owns the quality gate. | HF datasets |
| `models/` | Model definitions and MLflow registry access. | MLflow |
| `training/` | Orchestrates experiments, tuning, evaluation, promotion. | `data`, `models` |
| `serving/` | HTTP surface, request validation, instrumentation. | `models.registry` |
| `responsible/` | Fairness probe, explainers, report generation. | HTTP API only |
| `monitoring/` | Prometheus config, alert rules, provisioned Grafana dashboards. | `serving` metrics |

**Isolation rule:** `serving` imports from `models.registry` and nothing else in the
training half of the tree. `responsible/fairness.py` reaches the model **over HTTP against
the running service**, never by importing it. This is deliberate: it means the fairness
numbers describe the deployed system, including preprocessing and truncation, and are
therefore defensible under questioning.

### 2.3 Data flow

**Training path.** `amazon_polarity` (3.6M train / 400k test) is downloaded to
`data/raw/*.parquet`. The validation gate asserts schema, non-empty text, label balance
within tolerance, duplicate rate below threshold, and text-length distribution bounds; any
violation raises and fails the run. `preprocess` performs a seeded stratified subsample —
default 200,000 train / 25,000 validation / 25,000 test — concatenating `title` and
`content` into one field. The seed and all split sizes are logged as MLflow params.

**Training reference for drift.** `preprocess` also emits a reference distribution
(input-length histogram bins and the positive-class prior) which is logged as an MLflow
artifact beside the model. The serving layer loads it with the model. This coupling is what
makes the drift metric meaningful: the reference always matches the model actually serving.

**Serving path.** Request → pydantic validation → tokenise → forward pass → softmax →
response. In the same call, the API updates its rolling window of input lengths and
confidences and recomputes PSI against the reference. Response carries `model_version` so a
prediction can always be traced back to an MLflow run.

**Edge cases handled explicitly:** empty or whitespace-only text → 422; text exceeding the
configured character limit → 413; text longer than the model's 512-token window →
truncated, with a `truncated: true` flag in the response; batch containing a mix of valid
and invalid items → per-item results with per-item errors, HTTP 207-style semantics in the
body; model not yet loaded → 503 from `/ready` and from prediction endpoints; non-ASCII and
emoji input → preserved through tokenisation, covered by a unit test.

### 2.4 Technology choices and trade-offs

| Decision | Chosen | Alternatives considered | Rationale / trade-off |
|---|---|---|---|
| Serving framework | FastAPI | Flask, BentoML | Generates the OpenAPI spec the rubric requires for free; async I/O; pydantic validation is also the request-contract test. Flask would need extra libraries for the same. |
| Model | DistilBERT (+ TF-IDF/LogReg baseline) | BERT-base, RoBERTa, LinearSVC only | DistilBERT is ~40% smaller and ~60% faster than BERT-base at ~97% of its quality — the right point on the latency/accuracy curve for a 200 ms CPU budget. The classical baseline is kept because it makes the experiment comparison honest and gives CI a model it can load in milliseconds. |
| Tracking | MLflow + postgres | Weights & Biases, file backend | Named in the brief; postgres backend demonstrates a real backend store and survives container restarts, at the cost of one more service. |
| Metrics | `prometheus_client` in-process | Sidecar exporter, StatsD | Drift and confidence are functions of the prediction itself; exporting them from anywhere else would require shipping predictions to a second process for no benefit. |
| Drift measure | PSI on input length + class prior | KS test, KL divergence, Evidently | PSI has a conventional interpretation (< 0.1 stable, 0.1–0.2 moderate, > 0.2 significant) which justifies a concrete alert threshold rather than an arbitrary one. |
| Fairness benchmark | Equity Evaluation Corpus | Demographic groupby, Fairlearn on tabular attrs | Review datasets carry no demographic columns, so group-wise metrics are impossible. EEC probes the *model* with minimally-different sentence pairs, which is the established method for this exact task (Kiritchenko & Mohammad, 2018). |
| Explainability | SHAP (global) + LIME (local) | Attention weights, Integrated Gradients | The rubric's "Excellent" asks for multiple methods. Attention weights are contested as explanations; SHAP and LIME are defensible. |
| Orchestration | Docker Compose | Kubernetes, plain Docker | Required by the brief; Compose profiles let training and serving live in one file without a second orchestrator. |

**Scalability.** Single-replica CPU inference is the target. Horizontal scaling would work
(the API is stateless apart from the in-memory drift window, which would need moving to a
shared store) and this limitation is documented rather than engineered around.

**Cost.** Everything runs on a laptop. The largest cost is the initial dataset download.

**Complexity.** Six services is the deliberate ceiling. A feedback loop and retraining
scheduler were designed and rejected for this timeline (§9).

---

## 3. Implementation

### 3.1 Repository layout

```
sentiment-service/
├── README.md                    project overview, badges, quickstart, troubleshooting
├── ARCHITECTURE.md              system design (derived from §2 of this spec)
├── CONTRIBUTING.md              team roles, branching, commit conventions
├── pyproject.toml               deps, ruff, mypy, pytest, coverage config
├── requirements.txt             pinned, generated from pyproject
├── .gitignore  .env.example  Makefile
├── docker-compose.yml
├── docker/
│   ├── api.Dockerfile           multi-stage, non-root, HEALTHCHECK
│   └── training.Dockerfile
├── src/sentiment/
│   ├── config.py                pydantic-settings, 12-factor
│   ├── data/
│   │   ├── ingest.py            HF download → raw parquet
│   │   ├── validate.py          quality gate
│   │   └── preprocess.py        clean, split, drift reference
│   ├── models/
│   │   ├── baseline.py          TF-IDF + LogisticRegression
│   │   ├── transformer.py       DistilBERT fine-tune
│   │   └── registry.py          MLflow load / promote
│   ├── training/
│   │   ├── train.py             CLI entrypoint
│   │   ├── tune.py              Optuna, nested MLflow runs
│   │   └── evaluate.py          metrics, plots, promotion gate
│   ├── serving/
│   │   ├── app.py               FastAPI app + lifespan
│   │   ├── schemas.py           pydantic v2 with examples
│   │   ├── predictor.py         model wrapper
│   │   ├── metrics.py           Prometheus collectors
│   │   └── errors.py            typed error handlers
│   └── responsible/
│       ├── fairness.py          EEC probe over HTTP
│       ├── explain.py           SHAP + LIME
│       └── report.py            markdown/HTML report generation
├── tests/
│   ├── unit/  integration/  data_quality/  model/
├── monitoring/
│   ├── prometheus/prometheus.yml
│   ├── prometheus/alerts.yml
│   ├── alertmanager/alertmanager.yml
│   └── grafana/provisioning/{datasources,dashboards}/ + dashboards/*.json
├── docs/
│   ├── api.md  user-guide.md
│   └── superpowers/specs/
└── .github/workflows/  ci.yml  cd.yml
```

### 3.2 ML pipeline

Both model families train under a single MLflow experiment, `sentiment-amazon-polarity`.
Optuna sweeps `C` and `ngram_range` for the baseline, and learning rate, batch size, epoch
count, and warmup ratio for DistilBERT; each trial is a nested MLflow run.

Every run logs: params (including data seed and split sizes), metrics (accuracy, macro-F1,
precision/recall per class, ROC-AUC, p50/p95 inference latency, **fairness delta**), and
artifacts (model, tokenizer, drift reference, confusion matrix, ROC curve, PR curve, SHAP
summary plot).

`evaluate.py` implements the **promotion gate**: a run is promoted to registry stage
`Production` only if macro-F1 ≥ 0.92 **and** fairness delta ≤ 0.05. A model that is accurate
but biased cannot reach production. This single rule is what converts the Responsible AI
section from a report into an enforced system property.

### 3.3 Serving

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/predict` | `{text}` → `{label, score, confidence, model_version, truncated, latency_ms}` |
| POST | `/api/v1/predict/batch` | ≤ 64 texts, per-item success/error |
| POST | `/api/v1/explain` | `{text, method}` → token attributions |
| GET | `/api/v1/model/info` | version, metrics, fairness delta, trained-at |
| GET | `/health` | process liveness, no model dependency |
| GET | `/ready` | 200 only once a model is loaded |
| GET | `/metrics` | Prometheus exposition |
| GET | `/docs`, `/openapi.json` | auto-generated OpenAPI 3.1 |

`score` and `confidence` are distinct and must not be conflated: **`score` is P(positive)**,
in [0, 1], and is what the fairness probe compares across identity pairs;
**`confidence` is `max(score, 1 − score)`**, in [0.5, 1], and is what drives CX queue
ordering and the low-confidence metric. The EEC delta is always measured on `score`, because
a bias that flips a prediction from positive to negative leaves `confidence` unchanged.

Errors return `{error_code, message, request_id}`. `request_id` is generated per request in
middleware and included in structured logs, so a user-reported failure is traceable without
logging review text.

API versioning is by URL prefix (`/api/v1`), chosen over header-based negotiation because it
is visible in dashboards and trivially demonstrable in a demo.

### 3.4 Containerisation and orchestration

`api.Dockerfile` is two-stage: a builder installs dependencies into a virtualenv, and a slim
runtime stage copies only that venv and the source. The runtime image pins its base by
digest, creates and switches to an unprivileged user, and declares a `HEALTHCHECK` against
`/ready`.

`docker-compose.yml` defines six services — `api`, `mlflow`, `postgres`, `prometheus`,
`grafana`, `alertmanager` — each with a health check, and uses
`depends_on: {condition: service_healthy}` so startup order is enforced rather than hoped
for. Named volumes persist postgres data, MLflow artifacts, and Grafana state. A `training`
profile holds the one-shot pipeline container, so `docker compose up` starts only the
serving stack while `docker compose --profile training up` runs ingest → train → register.

### 3.5 Monitoring

```
sentiment_requests_total{endpoint,status,model_version}   counter
sentiment_request_duration_seconds{endpoint}              histogram
sentiment_predictions_total{label,model_version}          counter
sentiment_confidence                                      histogram
sentiment_low_confidence_total                            counter
sentiment_input_length_chars                              histogram
sentiment_drift_psi                                       gauge
sentiment_model_info{version,f1,fairness_delta}           gauge
```

Three Grafana dashboards, provisioned from files in the repo so they survive
`docker compose down -v` and are reviewable in a pull request:

1. **Service Health** — request rate, error rate, latency percentiles (RED method).
2. **Model Behaviour** — predicted class distribution vs. training prior, confidence
   histogram, low-confidence rate, drift PSI over time.
3. **Fairness & Explainability** — latest EEC deltas per identity group, promotion-gate
   status.

Alert rules, each carrying a `runbook_url` annotation pointing at a section of
`docs/user-guide.md`:

| Alert | Condition | Threshold rationale |
|---|---|---|
| `APIDown` | `up == 0` for 1m | Immediate. |
| `HighErrorRate` | 5xx ratio > 5% for 5m | Above expected client-error noise. |
| `HighLatencyP95` | p95 > 500 ms for 5m | 2.5× the 200 ms SLO, avoids flapping. |
| `PredictionSkew` | positive rate deviates > 20 pp from training prior for 15m | Wide enough to tolerate genuine traffic variation. |
| `DriftDetected` | `sentiment_drift_psi > 0.2` for 10m | Conventional PSI "significant shift" boundary. |
| `ModelNotReady` | `/ready` failing for 2m | Distinguishes a bad deploy from a crash. |

---

## 4. Testing & CI/CD

### 4.1 Test types

- **unit** (`tests/unit/`) — preprocessing, feature extraction, PSI computation, schema
  validation, error mapping.
- **integration** (`tests/integration/`) — every endpoint via `httpx.AsyncClient` against
  the real ASGI app; plus a Compose smoke test that brings the stack up and asserts
  `/ready`, a prediction, and that Prometheus has scraped a sample.
- **data_quality** (`tests/data_quality/`) — the validation gate's rules, exercised against
  both fixtures and real data: schema conformance, null/empty text, label balance,
  duplicate rate, length outliers, encoding.
- **model** (`tests/model/`) — behavioural rather than metric-threshold tests: known-positive
  and known-negative invariants, monotonicity under intensifiers, confidence calibration
  bounds, inference latency budget, and **the EEC fairness assertion as a hard failing
  test**.

### 4.2 Coverage

Target ≥ 80%, enforced by `--cov-fail-under=80` in the pytest configuration so it fails CI
rather than degrading silently. Coverage measures `src/sentiment/` only; notebooks are
excluded.

### 4.3 Pipelines

`ci.yml`, on push and pull request: ruff (lint + format check) → mypy → pytest with coverage
→ upload coverage report → build the API image → Trivy scan → Compose smoke test.
Transformer-training tests are marked `@pytest.mark.slow` and excluded from this path, so PR
feedback stays under roughly three minutes; they run on a nightly schedule.

`cd.yml`, on tag: rebuild and push the image to GHCR with the tag and `latest`.

### 4.4 Fairness as a gate

The fairness threshold appears in three places, and this repetition is intentional: as a
model-level success metric (§1.5), as the promotion gate in `evaluate.py` (§3.2), and as a
failing test in `tests/model/` (§4.1). A regression in fairness therefore blocks promotion
*and* breaks the build.

---

## 5. Responsible AI

### 5.1 Fairness

Amazon Polarity carries no demographic attributes, so group-wise fairness metrics cannot be
computed from the data. Instead the deployed model is probed with the **Equity Evaluation
Corpus** (Kiritchenko & Mohammad, 2018): template sentences that are identical except for a
substituted gendered name or word, or a name statistically associated with African-American
versus European-American identity.

`responsible/fairness.py` sends each pair to the running API and reports, per identity group:
mean sentiment score, mean pairwise delta, maximum delta, and a paired significance test.
The headline number is the maximum absolute delta across groups.

**Mitigation, applied if the threshold is exceeded:** counterfactual data augmentation
(name-swapping the training set so gendered and race-associated names appear in both
sentiment classes) and identity-term reweighting, followed by re-measurement. The
deliverable is a **before/after table** showing the delta shrinking, alongside the accuracy
cost of the mitigation — analysis without a mitigation attempt does not meet the bar.

### 5.2 Explainability

- **SHAP** — global feature importance over a test sample, producing a summary plot logged
  as an MLflow artifact and rendered in the Fairness dashboard.
- **LIME** — per-request local explanation, exposed live at `/api/v1/explain`, returning
  token-level attributions suitable for rendering.

Both are covered because the rubric's top band names multiple methods, and because they
answer different questions: SHAP explains the model, LIME explains a decision.

### 5.3 Privacy

Reviews are public but pseudonymous. The design specifies: PII scrubbing (emails, phone
numbers, URLs) at ingestion; **no request or response bodies in production logs** — only a
salted hash of the text, its length, and the request id; a documented retention limit on the
rolling drift window (counts and histogram bins only, never raw text); and `.env.example`
carrying no real secrets.

### 5.4 Ethics

Documented in `docs/` and presented: automated sentiment scoring used to *suppress* rather
than prioritise reviews would be a misuse, and the system is scoped to triage only. Known
failure modes are stated plainly — sarcasm and negation inversion, domain shift from
product reviews to other text, and the risk that non-native-English writing is scored more
negatively because fluency correlates with the training distribution's positive class. Each
is paired with a mitigation or an explicit acceptance with a monitoring hook.

---

## 6. Documentation

- **README.md** — badges (CI, coverage, licence), one-command quickstart, architecture
  thumbnail, endpoint examples with `curl`, troubleshooting section, links to the other docs.
- **ARCHITECTURE.md** — §2 of this spec, expanded, with the diagrams.
- **CONTRIBUTING.md** — the role table below, branch naming, commit message convention,
  PR checklist.
- **docs/api.md** — endpoint reference; the authoritative machine-readable spec is the
  generated `/openapi.json`.
- **docs/user-guide.md** — deployment, operation, dashboard walkthrough, and the alert
  runbooks referenced by `runbook_url`.

Code quality: type hints throughout, docstrings on public functions, ruff for lint and
format, mypy in CI.

---

## 7. Team roles

| | Owner | Primary directories |
|---|---|---|
| M1 | Data & Features | `src/sentiment/data/`, `tests/data_quality/` |
| M2 | Training & Experiments | `src/sentiment/models/`, `src/sentiment/training/`, `tests/model/` |
| M3 | Serving & Containers | `src/sentiment/serving/`, `docker/`, `docker-compose.yml` |
| M4 | Monitoring & CI/CD | `monitoring/`, `.github/workflows/`, `tests/integration/` |
| M5 | Responsible AI & Docs | `src/sentiment/responsible/`, `docs/`, `README.md`, `ARCHITECTURE.md` |

Ownership is by directory and is disjoint, so the git history demonstrates individual
contribution without anyone having to argue for it during the ±20% adjustment. Every member
reviews at least one PR per week outside their own area.

---

## 8. Schedule

**Week 1 — walking skeleton.** Repo scaffold, `docker-compose.yml` with all six services
healthy, ingest and validate working, FastAPI serving a *stub* model that returns random
labels, CI green from the first commit.

**Week 2 — real models.** Baseline and DistilBERT training, MLflow tracking and registry,
API loading the promoted model, Prometheus scraping real metrics.

**Week 3 — the graded surface.** Grafana dashboards and alert rules, EEC fairness probe,
SHAP and LIME, coverage raised to 80%.

**Week 4 — freeze and rehearse.** No new features. Documentation, slide deck, and two full
end-to-end rehearsals starting from a clean `git clone` on a machine that has never run the
project.

The week-1 walking skeleton is the highest-leverage item in this schedule. Integrating six
services is where the surprises live, and discovering them in week 4 is how live demos fail.

---

## 9. Deferred

**Feedback loop and continuous retraining** — a `POST /api/v1/feedback` endpoint, Redis, and
a scheduled retraining job. Designed and deliberately deferred: it roughly doubles the
service count for a narrative benefit that the drift metric already delivers, and it is the
component most likely to be half-finished at the deadline. If week 3 finishes early, the
feedback *endpoint* alone (persisting to postgres, no scheduler) is the cheapest increment.

**Multi-language support** — noted in §1.6.

---

## 10. Rubric coverage

| Rubric section | Weight | Where addressed |
|---|---|---|
| Problem definition & requirements | 10% | §1 |
| System design & architecture | 15% | §2 |
| Implementation — ML pipeline | 15% | §3.2 |
| Implementation — Deployment | 15% | §3.3, §3.4 |
| Implementation — Monitoring | 10% | §3.5 |
| Testing & CI/CD | 15% | §4 |
| Responsible AI | 10% | §5 |
| Documentation | 10% | §6 |
| Individual contribution (±20%) | — | §7 |
