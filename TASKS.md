# TASKS — sentiment-service

Task board for the DDM501 Final Project. One line per assignable unit of work.

- **Detailed steps for Week 1** live in [`docs/superpowers/plans/2026-08-09-walking-skeleton.md`](docs/superpowers/plans/2026-08-09-walking-skeleton.md) — the `Plan` column points at the task number there.
- **Design rationale** for everything lives in [`docs/superpowers/specs/2026-08-09-sentiment-service-design.md`](docs/superpowers/specs/2026-08-09-sentiment-service-design.md) — the `Spec` column points at the section.
- Weeks 2-4 are listed at task granularity here; their step-by-step plans get written at the start of each week.

**Status:** `TODO` · `WIP` · `REVIEW` (PR open) · `DONE` (merged, CI green)

## Owners

Fill in real names before the first PR — `CONTRIBUTING.md` and the ±20% individual
contribution adjustment both depend on this table.

| ID | Name | Area | Owns |
|---|---|---|---|
| M1 | *(fill in)* | Data & Features | `src/sentiment/data/`, `tests/data/` |
| M2 | *(fill in)* | Training & Experiments | `src/sentiment/models/`, `src/sentiment/training/`, `scripts/train_model.py`, `scripts/validate_model.py`, `tests/model/` |
| M3 | *(fill in)* | Serving & Containers | `src/sentiment/serving/`, `Dockerfile`, `docker-compose.yml` |
| M4 | *(fill in)* | Monitoring & CI/CD | `prometheus/`, `grafana/`, `alertmanager/`, `scripts/load_test.py`, `.github/workflows/`, `tests/integration/` |
| M5 | *(fill in)* | Responsible AI & Docs | `src/sentiment/responsible/`, `scripts/run_fairness_probe.py`, `docs/`, `README.md`, `ARCHITECTURE.md` |

Directory ownership is disjoint on purpose: it keeps merge conflicts rare and makes
the git history evidence of who did what, without anyone having to argue for it.

## Working agreement

- Branch: `<initials>/<short-description>` — e.g. `nt/data-quality-gate`.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `ci:`).
- Every PR: tests pass, coverage ≥ 80%, `make lint` and `make typecheck` clean.
- Every member reviews at least one PR per week **outside** their own area.
- Nobody merges their own PR.

---

## Week 1 — Walking skeleton

**Goal:** every service in `docker-compose.yml` talking to every other service, serving predictions from a stub model, CI green. No real model yet — that is deliberate.

**Why this order:** integrating six services is where the surprises live. Teams that leave integration to Week 4 are the teams whose live demo fails.

| ID | Task | Owner | Depends on | Plan | Status |
|---|---|---|---|---|---|
| W1-01 | Scaffold: `pyproject.toml`, `requirements*.txt`, `.flake8`, `Makefile`, `.env.example`, `conftest.py`, `config.py` + tests | M3 | — | Task 1 | TODO |
| W1-02 | Data ingestion: `amazon_polarity` → normalised Parquet | M1 | W1-01 | Task 2 | TODO |
| W1-03 | Data quality gate: schema, empties, duplicates, label balance — **fails the run** | M1 | W1-02 | Task 3 | TODO |
| W1-04 | Splits + drift reference (seeded, stratified, logged as an artifact) | M1 | W1-03 | Task 4 | TODO |
| W1-05 | `Predictor` protocol + deterministic `StubPredictor` | M3 | W1-01 | Task 5 | TODO |
| W1-06 | Prometheus collectors (`http_*` / `ml_*`) + PSI `DriftTracker` | M4 | W1-04 | Task 6 | TODO |
| W1-07 | FastAPI app: schemas, typed errors, `MetricsMiddleware`, `/health` `/ready` `/metrics` | M3 | W1-05, W1-06 | Task 7 | TODO |
| W1-08 | Endpoints: `/predict`, `/predict/batch`, `/model/info` + ML instrumentation | M3 | W1-07 | Task 8 | TODO |
| W1-09 | `Dockerfile` — multi-stage, non-root, `HEALTHCHECK` on `/ready` | M3 | W1-08 | Task 9 | TODO |
| W1-10 | `docker-compose.yml` (6 services, health checks) + `prometheus/` + `grafana/` + `alertmanager/` + smoke test | M4 | W1-09 | Task 10 | TODO |
| W1-11 | CI: lint / type-check → test matrix → container → `ci-status` | M4 | W1-10 | Task 11 | TODO |
| W1-12 | `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `docs/user-guide.md`, `docs/TESTING_STRATEGY.md` | M5 | W1-10 | Task 12 | TODO |

**Parallelism.** W1-01 blocks everything, so do it first and merge it fast — one person, one sitting. After that M1 runs W1-02→04 while M3 runs W1-05→08, independently. M4 starts W1-06 as soon as W1-04 lands. M5 has no code dependency and should start W1-12 immediately, filling in details as the other tracks land.

**M5 in week 1 is deliberately light** — use the slack to write the problem-definition and requirements sections (Spec §1), which are 10% of the grade and need no code.

### Week 1 gate — do not start Week 2 until all of these hold

- [ ] `docker compose down -v && docker compose up -d --build` → six healthy services from scratch
- [ ] `curl -X POST localhost:8000/api/v1/predict -H 'content-type: application/json' -d '{"text":"great"}'` returns a full prediction
- [ ] Prometheus: `up{job="sentiment-api"} == 1`, eight alert rules loaded
- [ ] Grafana reachable, Prometheus datasource provisioned from files
- [ ] `/metrics` exposes both `http_*` and `ml_*` families
- [ ] `pytest -m "not slow"` passes, coverage ≥ 80%
- [ ] `make lint` and `make typecheck` clean
- [ ] CI green on `main`
- [ ] Required files exist: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/`
- [ ] **Every team member has at least one commit**

---

## Week 2 — Real models

**Goal:** the stub is gone. A trained model is promoted through the MLflow registry and served.

| ID | Task | Owner | Depends on | Spec | Status |
|---|---|---|---|---|---|
| W2-01 | MLflow registry client: load promoted model, promote a run | M2 | W1-11 | §3.2 | TODO |
| W2-02 | Baseline model: TF-IDF + LogisticRegression | M2 | W1-04 | §2.4 | TODO |
| W2-03 | DistilBERT fine-tune, satisfying the `Predictor` protocol | M2 | W1-05 | §2.4 | TODO |
| W2-04 | `scripts/train_model.py` — ingest → validate → preprocess → fit → evaluate | M2 | W2-02, W2-03 | §3.2 | TODO |
| W2-05 | Optuna sweep, one nested MLflow run per trial | M2 | W2-04 | §3.2 | TODO |
| W2-06 | `evaluate.py`: metrics, plots, and the **fairness-gated promotion rule** | M2 | W2-04 | §3.2, §4.4 | TODO |
| W2-07 | `scripts/validate_model.py` — quality gate callable from CI | M2 | W2-06 | §4.3 | TODO |
| W2-08 | Swap `StubPredictor` → `TransformerPredictor` in `_lifespan`; load drift reference from the MLflow artifact | M3 | W2-01, W2-03 | §2.3 | TODO |
| W2-09 | Model behaviour tests: known-positive/negative, calibration bounds, latency budget | M2 | W2-08 | §4.1 | TODO |
| W2-10 | Add `training` profile to compose; add MLflow scrape target | M4 | W2-04 | §3.4 | TODO |
| W2-11 | Nightly workflow for `@pytest.mark.slow` transformer tests | M4 | W2-09 | §4.3 | TODO |
| W2-12 | Record baseline-vs-transformer comparison in `docs/` (the experiment story) | M5 | W2-05 | §3.2 | TODO |

**W2-08 is the payoff of the walking skeleton.** It should touch only the lifespan
binding and the drift-reference source. If it turns into a large diff, something in
Week 1 was built wrong — stop and fix the boundary rather than working around it.

### Week 2 gate

- [ ] MLflow shows ≥ 10 runs across both model families, with params, metrics, and artifacts
- [ ] A model sits in registry stage `Production`, promoted by the gate rather than by hand
- [ ] `/api/v1/model/info` reports the real version, F1, and fairness delta
- [ ] Macro-F1 ≥ 0.92 on the held-out test split
- [ ] p95 latency < 200 ms on CPU
- [ ] Coverage still ≥ 80%; CI still green

---

## Week 3 — The graded surface

**Goal:** the sections that carry 20% of the grade but are usually left to the last minute.

| ID | Task | Owner | Depends on | Spec | Status |
|---|---|---|---|---|---|
| W3-01 | Import Lab 4's `system_dashboard.json`, adapt to this service | M4 | W2-08 | §3.5 | TODO |
| W3-02 | Import Lab 4's `ml_dashboard.json`; add confidence, class-skew, drift panels | M4 | W2-08 | §3.5 | TODO |
| W3-03 | New Fairness & Explainability dashboard | M4 | W3-05 | §3.5 | TODO |
| W3-04 | `scripts/load_test.py` — generate traffic so panels and alerts have data | M4 | W3-01 | §3.5 | TODO |
| W3-05 | EEC fairness probe over HTTP; export `ml_fairness_max_delta` | M5 | W2-08 | §5.1 | TODO |
| W3-06 | `fairness_alerts.yml` — `FairnessRegression` at the promotion threshold | M4 | W3-05 | §3.5 | TODO |
| W3-07 | Fairness **mitigation**: counterfactual augmentation + reweighting, re-measure | M5 | W3-05 | §5.1 | TODO |
| W3-08 | **Before/after fairness table** — the artifact that earns the top band | M5 | W3-07 | §5.1 | TODO |
| W3-09 | SHAP global importance, logged as an MLflow artifact | M5 | W2-08 | §5.2 | TODO |
| W3-10 | LIME local explanations + `POST /api/v1/explain` endpoint | M5, M3 | W2-08 | §5.2 | TODO |
| W3-11 | Ethics & privacy write-up: PII scrubbing, no-body logging, failure modes | M5 | — | §5.3, §5.4 | TODO |
| W3-12 | Raise coverage to ≥ 80% across all four test types; fill gaps | all | — | §4 | TODO |
| W3-13 | Alert runbooks in `docs/user-guide.md`, one `##` per alert, anchors verified | M5 | W3-06 | §6 | TODO |

**W3-08 is the highest-value single item in this week.** A fairness *analysis*
scores "Good". A before/after table showing the delta shrinking, with the accuracy
cost stated, is what "Excellent" describes. Budget time for it.

### Week 3 gate

- [ ] Three dashboards provisioned from files, surviving `docker compose down -v`
- [ ] Every alert fires correctly under `scripts/load_test.py` — test at least `DriftDetected` and `HighLatencyP95` for real
- [ ] `POST /api/v1/explain` returns token attributions live
- [ ] Fairness before/after table exists, with the accuracy cost stated
- [ ] Coverage ≥ 80% with all four test types present
- [ ] Every runbook anchor resolves (the `grep` check in plan Task 12, Step 5)

---

## Week 4 — Freeze and rehearse

**Feature freeze on day 1 of this week.** Anything not merged by then ships as
"future work" in the README. This is not negotiable — the demo is 15% of the
presentation grade and rehearsal time is what protects it.

| ID | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| W4-01 | Replace `github.com/OWNER` in every `runbook_url` with the real repo | M4 | — | TODO |
| W4-02 | README polish: badges, quickstart, `curl` examples with real output, troubleshooting | M5 | — | TODO |
| W4-03 | `ARCHITECTURE.md` final pass; diagrams match what actually runs | M5 | — | TODO |
| W4-04 | `CONTRIBUTING.md` with real names and per-member contribution summary | M5 | — | TODO |
| W4-05 | OpenAPI examples on every schema; `docs/api.md` cross-check | M3 | — | TODO |
| W4-06 | Slide deck (15-20 min): problem → architecture → deep dive → responsible AI → demo | all | — | TODO |
| W4-07 | **Rehearsal 1**: clean `git clone` on a machine that has never run this | all | W4-01..05 | TODO |
| W4-08 | Fix everything rehearsal 1 broke | all | W4-07 | TODO |
| W4-09 | **Rehearsal 2**: clean clone again, timed, every member speaks | all | W4-08 | TODO |
| W4-10 | Q&A prep: each member writes 3 likely questions on their own area and answers them | all | W4-09 | TODO |

**W4-07 must run on hardware that has never built this project.** Docker layer
caches and leftover volumes hide broken configuration, and the grader's machine
will not have them.

### Submission checklist

- [ ] Repo public, or instructor added as collaborator
- [ ] All five members have meaningful, spread-out commits
- [ ] `.gitignore` excludes `data/`, `models/`, `mlruns/`, `.venv/`
- [ ] Required files present: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/`
- [ ] CI green on `main` at the submitted commit
- [ ] Slides finalised, demo rehearsed twice end-to-end

---

## Risk register

| Risk | Trigger | Mitigation |
|---|---|---|
| Live demo fails | Untested on a clean machine | W4-07 and W4-09, on hardware that never built the project |
| Coverage misses 80% | Left to Week 3 | `--cov-fail-under=80` is on from W1-01, so it can never silently rot |
| Fairness section thin | Treated as a report, not a system property | Promotion gate (W2-06) + failing test (W2-09) + alert (W3-06) |
| Uneven contribution | One person builds everything | Disjoint directory ownership; cross-area PR review each week |
| DistilBERT too slow on CPU | Discovered in Week 4 | Latency budget is a test from W2-09; baseline model is always a fallback |
| Dataset download blocks CI | 3.6M rows fetched on every run | Subsample is seeded and cached; CI uses fixtures, not the real download |
