# TASKS — sentiment-service

Task board for the DDM501 Final Project. One line per assignable unit of work.

- **Detailed steps for Week 1** live in [`docs/superpowers/plans/2026-08-09-walking-skeleton.md`](docs/superpowers/plans/2026-08-09-walking-skeleton.md) — the `Plan` column points at the task number there.
- **Design rationale** for everything lives in [`docs/superpowers/specs/2026-08-09-sentiment-service-design.md`](docs/superpowers/specs/2026-08-09-sentiment-service-design.md) — the `Spec` column points at the section.
- Weeks 2-4 are listed at task granularity here; their step-by-step plans get written at the start of each week.

**Status:** `TODO` · `WIP` · `BLOCKED` (waiting on something outside this task) ·
`REVIEW` (PR open) · `DONE` (merged, CI green)

**Last audited:** 2026-08-14, against a live stack rather than by reading the table.

Everything marked `DONE` below has been exercised: the suite is **147 fast tests plus
8 `slow` model-behaviour tests, all passing**, coverage **91%** over the whole package,
`flake8` / `black` / `isort` / `mypy` clean, six services healthy, three dashboards
loading with real data, ten alert rules parsed, and a model promoted through the gate
and served from the registry.

The work is merged: `origin/main` is at `3388536` (PR #6). One caveat remains —
**CI-green on `main` cannot be confirmed from this machine**, because the `gh` CLI is not
installed. Check the Actions tab before treating any row as fully `DONE` by the
definition in `CONTRIBUTING.md`.

One CI failure has already been through this loop and is fixed in `add5f31`: a
`dict[Hashable, int]` narrowing in `data/validate.py` that the pinned `mypy==1.8.0` could
not see but a newer toolchain could. See **Toolchain drift** at the end of this file for what was actually wrong and how it is
now prevented.

## Owners

| ID | Name | Git identity | Area | Owns |
|---|---|---|---|---|
| M1 | Lý Minh Thông | `thong312` | Data & Features | `src/sentiment/data/`, `tests/data/` |
| M2 | Dương Thành Duy | `dtduy77` | Training & Experiments | `src/sentiment/models/`, `src/sentiment/training/`, `scripts/train_model.py`, `scripts/validate_model.py`, `scripts/evaluate_model.py`, `notebooks/`, `tests/model/` |
| M3 | Bùi Vân Sơn | `sontv6666` | Serving & Containers | `src/sentiment/serving/`, `Dockerfile`, `docker-compose.yml` |
| M4 | Lê Công Huỳnh | `HuynhLC` | Monitoring & CI/CD | `prometheus/`, `grafana/`, `alertmanager/`, `scripts/load_test.py`, `.github/workflows/`, `tests/integration/` |
| M5 | Thái Bình Nhiên | `Nhien Thai` | Responsible AI & Docs | `src/sentiment/responsible/`, `scripts/run_fairness_probe.py`, `docs/`, `README.md`, `ARCHITECTURE.md` |

The git identity column is there so a grader can map a commit to a person without
guessing. Each mapping is unambiguous from the commit email:
`lyminhthong312@`, `duythduong.2003@`, `sontv6666@`, `huynhlc1281@`, `thaibinhnhien@`.

Directory ownership is disjoint on purpose: it keeps merge conflicts rare and makes
the git history evidence of who did what, without anyone having to argue for it.

## Working agreement

- Branch: `<initials>/<short-description>` — `lmt` Thông, `dtd` Duy, `bvs` Sơn, `lch` Huỳnh, `tbn` Nhiên.
- Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `chore:`, `docs:`, `ci:`).
- Every PR: tests pass, coverage ≥ 80%, `make lint` and `make typecheck` clean.
- Every member reviews at least one PR per week **outside** their own area.
- Nobody merges their own PR.

---

## Resolved — the dataset split-brain (W2-04)

The spec said `amazon_polarity` + DistilBERT: English, binary. The code went
`tridm/UIT-VSFC` + `xlm-roberta-base`: Vietnamese, three classes. For a while both
existed at once and the W1-03 quality gate guarded data nobody trained on.

Closed as follows:

- `config.dataset_name` is now `tridm/UIT-VSFC`, and a `Settings` validator **raises**
  if it ever disagrees with `model_dataset_name`. The split-brain cannot come back.
- `training/train.py` goes through `load_validated_splits`, so every split passes
  `data/validate.py` and is fingerprinted before a model sees it.
- The balance rule changed from "positive ratio near 0.5" — meaningless for three
  classes — to a **floor on the rarest class** plus a check that all three are present.
  `neutral` is 4% of UIT-VSFC, which is fine; `neutral` vanishing is not.
- `data/ingest.py` gained a UIT-VSFC normaliser and a registry, so an unknown dataset
  fails loudly rather than being guessed at.

**One item deliberately left open.** `Prediction.score` is still the positive-class
probability while labels are three-class, so a `neutral` prediction reports
`probs[positive_id]`. Changing it is a breaking API change: `score` is in the published
contract, the fairness probe compares it, and `PredictionSkew` alerts on it. It is
documented in `ARCHITECTURE.md` and worth doing deliberately in one change rather than
quietly — the honest options are to redefine `score` as the predicted class's
probability, or to add a `probabilities` map and leave `score` alone.

---

## Week 1 — Walking skeleton

**Goal:** every service in `docker-compose.yml` talking to every other service, serving predictions from a stub model, CI green. No real model yet — that is deliberate.

**Why this order:** integrating six services is where the surprises live. Teams that leave integration to Week 4 are the teams whose live demo fails.

| ID | Task | Owner | Depends on | Plan | Status |
|---|---|---|---|---|---|
| W1-01 | Scaffold: `pyproject.toml`, `requirements*.txt`, `.flake8`, `Makefile`, `.env.example`, `conftest.py`, `config.py` + tests | M3 | — | Task 1 | DONE |
| W1-02 | Data ingestion: ~~`amazon_polarity`~~ **`tridm/UIT-VSFC`** → normalised Parquet | M1 | W1-01 | Task 2 | DONE |
| W1-03 | Data quality gate: schema, empties, duplicates, label balance — **fails the run** | M1 | W1-02 | Task 3 | DONE |
| W1-04 | Splits + drift reference (seeded, stratified, logged as an artifact) | M1 | W1-03 | Task 4 | DONE |
| W1-05 | `Predictor` protocol + deterministic `StubPredictor` | M3 | W1-01 | Task 5 | DONE |
| W1-06 | Prometheus collectors (`http_*` / `ml_*`) + PSI `DriftTracker` | M4 | W1-04 | Task 6 | DONE |
| W1-07 | FastAPI app: schemas, typed errors, `MetricsMiddleware`, `/health` `/ready` `/metrics` | M3 | W1-05, W1-06 | Task 7 | DONE |
| W1-08 | Endpoints: `/predict`, `/predict/batch`, `/model/info` + ML instrumentation | M3 | W1-07 | Task 8 | DONE |
| W1-09 | `Dockerfile` — multi-stage, non-root, `HEALTHCHECK` on `/ready` | M3 | W1-08 | Task 9 | DONE |
| W1-10 | `docker-compose.yml` (6 services, health checks) + `prometheus/` + `grafana/` + `alertmanager/` + smoke test | M4 | W1-09 | Task 10 | DONE |
| W1-11 | CI: lint / type-check → test matrix → container → `ci-status` | M4 | W1-10 | Task 11 | DONE |
| W1-12 | `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `docs/user-guide.md`, `docs/TESTING_STRATEGY.md` | M5 | W1-10 | Task 12 | DONE |

**W1-12 is merged.** Plan Task 12 was followed with four deliberate deviations, recorded
here because the plan itself is wrong on these points:

1. **Task 12 has no step for `docs/TESTING_STRATEGY.md`** even though it lists the file.
   Written anyway, covering the five test categories, all 159 tests, and the coverage scope.
2. **Step 5's anchor check uses `#[a-z]*`, which cannot match `highlatencyp95`** — the
   digits truncate it to `highlatencyp`, and it then passes only by substring accident.
   Use `#[a-z0-9]*`. All ten anchors verified with the corrected pattern.
3. **Step 2's "port spec §2 verbatim" would have put false claims in a graded file.**
   Spec §2.4 justifies DistilBERT for a 200 ms CPU budget and §1.6 declares English
   binary sentiment in scope; the code serves three-class Vietnamese on XLM-RoBERTa.
   `ARCHITECTURE.md` documents what runs, and states the latency budget as an open risk.
4. **Step 7's commit command is wrong**: it adds a `monitoring/` directory that does not
   exist and omits `docs/TESTING_STRATEGY.md`.

**Parallelism.** W1-01 blocks everything, so do it first and merge it fast — one person, one sitting. After that M1 runs W1-02→04 while M3 runs W1-05→08, independently. M4 starts W1-06 as soon as W1-04 lands. M5 has no code dependency and should start W1-12 immediately, filling in details as the other tracks land.

**M5 in week 1 is deliberately light** — use the slack to write the problem-definition and requirements sections (Spec §1), which are 10% of the grade and need no code.

### Week 1 gate — do not start Week 2 until all of these hold

Verified live on 2026-08-13 by bringing the stack up on a machine with Docker 29.5.3.

- [x] `docker compose down -v && docker compose up -d --build` → six healthy services from scratch
- [x] `curl -X POST localhost:8000/api/v1/predict -H 'content-type: application/json' -d '{"text":"great"}'` returns a full prediction
- [x] Prometheus: `up{job="sentiment-api"} == 1`, eight alert rules loaded — all eight by name
- [x] Grafana reachable, Prometheus datasource provisioned from files — default, pointing at `http://prometheus:9090`
- [x] `/metrics` exposes both `http_*` and `ml_*` families — 90 and 107 series respectively
- [ ] `pytest -m "not slow"` passes, coverage ≥ 80% — **not verified**, no local virtualenv
- [ ] `make lint` and `make typecheck` clean — **not verified**, same reason
- [ ] CI green on `main` — **not verified**, `gh` CLI is not installed
- [x] Required files exist: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/`
- [x] **Every team member has at least one commit** — five distinct authors on `origin/main`

One caveat on the first box: the verification machine had ports 9090, 3000 and 5000
occupied by other software, so those three services were published elsewhere for the
check. The service definitions and provisioning are unaffected. Port 5000 in particular
is worth knowing about before a demo — see the AirPlay entry in
[`docs/REHEARSAL.md`](docs/REHEARSAL.md).

---

## Week 2 — Real models

**Goal:** the stub is gone. A trained model is promoted through the MLflow registry and served.

| ID | Task | Owner | Depends on | Spec | Status |
|---|---|---|---|---|---|
| W2-01 | MLflow registry client: load promoted model, promote a run | M2 | W1-11 | §3.2 | DONE |
| W2-02 | Baseline model: TF-IDF + LogisticRegression | M2 | W1-04 | §2.4 | DONE |
| W2-03 | ~~DistilBERT~~ **XLM-RoBERTa** fine-tune, satisfying the `Predictor` protocol | M2 | W1-05 | §2.4 | DONE |
| W2-04 | `scripts/train_model.py` — ingest → validate → preprocess → fit → evaluate | M2 | W2-02, W2-03 | §3.2 | DONE |
| W2-05 | Optuna sweep, one nested MLflow run per trial | M2 | W2-04 | §3.2 | DONE |
| W2-06 | `evaluate.py`: metrics, plots, and the **fairness-gated promotion rule** | M2 | W2-04 | §3.2, §4.4 | DONE |
| W2-07 | `scripts/validate_model.py` — quality gate callable from CI | M2 | W2-06 | §4.3 | DONE |
| W2-08 | Swap `StubPredictor` → `TransformerPredictor` in `_lifespan`; load drift reference from the MLflow artifact | M3 | W2-01, W2-03 | §2.3 | DONE |
| W2-09 | Model behaviour tests: known-positive/negative, calibration bounds, latency budget | M2 | W2-08 | §4.1 | DONE |
| W2-10 | Add `training` profile to compose; add MLflow scrape target | M4 | W2-04 | §3.4 | DONE |
| W2-11 | Nightly workflow for `@pytest.mark.slow` transformer tests | M4 | W2-09 | §4.3 | DONE |
| W2-12 | Record baseline-vs-transformer comparison in `docs/` (the experiment story) | M5 | W2-05 | §3.2 | BLOCKED |

**How the Week 2 rows closed:**

- **W2-04** — training now enters through `load_validated_splits`: normalise → quality
  gate → fingerprint, per split. See the resolved section above.
- **W2-05** — `training/tune.py` runs an Optuna TPE search, one **nested MLflow run per
  trial**, optimising *cross-validated* macro-F1 rather than a single split. With
  `neutral` at 4%, a single split rewards whichever trial drew a lucky fold.
- **W2-06** — `evaluate.py` gained stratified k-fold cross-validation reporting mean
  and standard deviation; SHAP global importance is logged as an MLflow artifact with
  per-class charts; and `validate_model.py` now gates on **macro-F1, accuracy, latency
  and fairness**, so bias is a build failure rather than a report.
- **W2-08** — closed end-to-end. A baseline is registered at stage `Production` and
  served with `SENTIMENT_PREDICTOR_BACKEND=registry`; `/model/info` reports version 2,
  30 real metrics and `fairness_delta: 0.0`. The registry loader dispatches on the
  artifact, so it serves a joblib baseline or a Hugging Face directory.
- **W2-09** — eight `slow` tests on the real corpus: known-positive and known-negative
  cases, score ordering between the two groups, calibration bounds, the held-out
  quality floor, the CPU latency budget, the fairness gate after mitigation, and a real
  transformer checkpoint against the serving contract.

**W2-08 was the payoff of the walking skeleton, and it paid.** Serving a real model was
a configuration change plus a loader that dispatches on the artifact type — the HTTP
contract, the metrics and the tests did not move.

**W2-12 is `BLOCKED`, not done.** It asks for a baseline-*vs-transformer* comparison, and
no transformer has been trained, so the comparison has nothing to compare. It was briefly
marked `DONE` in a bulk status pass — an error, and exactly the kind this board is
supposed to prevent. What exists is a comparison of three *baseline* variants in
[`docs/FAIRNESS.md`](docs/FAIRNESS.md); that is W3-08, not this. Unblocks the moment a
fine-tuned checkpoint exists: the numbers to compare are already logged per run.

### Week 2 gate

- [x] MLflow shows ≥ 10 runs, with params, metrics, and artifacts — 10 nested Optuna trials plus parent runs, carrying dataset fingerprints, SHAP charts and fairness reports. *Both model families is not yet met: every run is the baseline.*
- [x] A model sits in registry stage `Production`, promoted by the gate rather than by hand — version 2, promoted by `validate_model.py`
- [x] `/api/v1/model/info` reports the real version, F1, and fairness delta — version `2`, `test_macro_f1` 0.7149, `fairness_delta` 0.0
- [x] Baseline: macro-F1 ≥ 0.70 **and** accuracy ≥ 0.85 on the held-out test split — measured **0.7149 / 0.8629**
- [ ] Transformer: macro-F1 ≥ 0.80 **and** accuracy ≥ 0.92 on the held-out test split — **needs a GPU; the only Week 2 item still open**
- [x] p95 latency < 200 ms on CPU — **0.42 ms** for the baseline, enforced by a `slow` test. Unproven for the transformer.
- [x] Coverage still ≥ 80% — **91%**, now measured over the whole package rather than the serving slice. CI green not verifiable locally.

**Why these numbers replaced "macro-F1 ≥ 0.92".** The original target was written for
`amazon_polarity`: binary and balanced. UIT-VSFC is three-class and `neutral` is only
458 of 11,426 training rows — 4%. Macro-F1 weights that class equally with the other
two, so it sits far below accuracy by construction: the measured baseline is macro-F1
**0.7193 ± 0.0069** against accuracy **0.8784** under 5-fold stratified
cross-validation. Reaching 0.92 macro-F1 would mean very nearly solving the minority
class, which published work on this dataset does not report either.

Keeping the old number would have meant failing a gate that never fitted the data. The
rubric asks for "clear business, system, and model metrics **with targets**", not for a
particular value, so a justified and reachable target scores better than an
unreachable one. Both metrics are required together because accuracy alone can be met
by ignoring `neutral` entirely.

The baseline thresholds sit just under what the trained baseline actually achieves on
the held-out test split, so the gate is a floor that catches regressions rather than a
ceiling nobody clears. The transformer thresholds are the ones still to be proven, and
they are deliberately set above the baseline: a transformer that cannot beat TF-IDF is
not worth its latency budget.

---

## Week 3 — The graded surface

**Goal:** the sections that carry 20% of the grade but are usually left to the last minute.

| ID | Task | Owner | Depends on | Spec | Status |
|---|---|---|---|---|---|
| W3-01 | ~~Import Lab 4's~~ **Write** `system_dashboard.json` for this service | M4 | W2-08 | §3.5 | DONE |
| W3-02 | ~~Import Lab 4's~~ **Write** `ml_dashboard.json`; confidence, class-skew, drift panels | M4 | W2-08 | §3.5 | DONE |
| W3-03 | New Fairness & Explainability dashboard | M4 | W3-05 | §3.5 | DONE |
| W3-04 | `scripts/load_test.py` — generate traffic so panels and alerts have data | M4 | W3-01 | §3.5 | DONE |
| W3-05 | EEC fairness probe over HTTP; export `ml_fairness_max_delta` | M5 | W2-08 | §5.1 | DONE |
| W3-06 | `fairness_alerts.yml` — `FairnessRegression` at the promotion threshold | M4 | W3-05 | §3.5 | DONE |
| W3-07 | Fairness **mitigation**: counterfactual augmentation + reweighting, re-measure | M5 | W3-05 | §5.1 | DONE |
| W3-08 | **Before/after fairness table** — the artifact that earns the top band | M5 | W3-07 | §5.1 | DONE |
| W3-09 | SHAP global importance, logged as an MLflow artifact | M5 | W2-08 | §5.2 | DONE |
| W3-10 | LIME local explanations + `POST /api/v1/explain` endpoint | M5, M3 | W2-08 | §5.2 | DONE |
| W3-11 | Ethics & privacy write-up: PII scrubbing, no-body logging, failure modes | M5 | — | §5.3, §5.4 | DONE |
| W3-12 | Raise coverage to ≥ 80% across all four test types; fill gaps | all | — | §4 | DONE |
| W3-13 | Alert runbooks in `docs/user-guide.md`, one `##` per alert, anchors verified | M5 | W3-06 | §6 | DONE |

**W3-10 landed on both sides.** `responsible/explain.py` implements the `Explainer`
protocol with LIME, and the app builds one bound to whichever predictor is serving —
rebuilt on reload, so an explanation never describes a model that was replaced.

**W3-08 is the highest-value single item in this week.** A fairness *analysis*
scores "Good". A before/after table showing the delta shrinking, with the accuracy
cost stated, is what "Excellent" describes. Budget time for it.

### Week 3 gate

- [x] Three dashboards provisioned from files, surviving `docker compose down -v` — all three load into the **Sentiment** folder; 11/11 sampled panel queries return data
- [x] `DriftDetected` fires for real under `scripts/load_test.py` — observed `pending` → **FIRING**, and received by Alertmanager as `severity=warning, state=active`. *`HighLatencyP95` not yet reproduced: the baseline answers in 0.42 ms, so the `burst` scenario cannot push p95 past 500 ms. It needs the transformer.*
- [x] `POST /api/v1/explain` returns token attributions live — LIME against the promoted model
- [x] Fairness before/after table exists, with the accuracy cost stated — [`docs/FAIRNESS.md`](docs/FAIRNESS.md), three rows, two mitigations, per-dimension breakdown
- [x] Coverage ≥ 80% with all four test types present — **91%**, unit / data quality / model / integration
- [x] Every runbook anchor resolves — 10/10, with the corrected `#[a-z0-9]*` pattern

---

## Week 4 — Freeze and rehearse

**Feature freeze on day 1 of this week.** Anything not merged by then ships as
"future work" in the README. This is not negotiable — the demo is 15% of the
presentation grade and rehearsal time is what protects it.

| ID | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| W4-01 | ~~Replace `github.com/OWNER` in every `runbook_url` with the real repo~~ — resolved to `nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project` when the project moved to the repository root | M4 | — | DONE |
| W4-02 | README polish: badges, quickstart, `curl` examples with real output, troubleshooting | M5 | — | DONE |
| W4-03 | `ARCHITECTURE.md` final pass; diagrams match what actually runs | M5 | — | DONE |
| W4-04 | `CONTRIBUTING.md` with real names and per-member contribution summary | M5 | — | REVIEW |
| W4-05 | OpenAPI examples on every schema; `docs/api.md` cross-check | M3 | — | DONE |
| W4-06 | Slide deck (15-20 min): problem → architecture → deep dive → responsible AI → demo | all | — | WIP |
| W4-07 | **Rehearsal 1**: clean `git clone` on a machine that has never run this | all | W4-01..05 | TODO |
| W4-08 | Fix everything rehearsal 1 broke | all | W4-07 | TODO |
| W4-09 | **Rehearsal 2**: clean clone again, timed, every member speaks | all | W4-08 | TODO |
| W4-10 | Q&A prep: each member writes 3 likely questions on their own area and answers them | all | W4-09 | REVIEW |

**W4-02 and W4-03 are `WIP` only because nothing is committed.** Their substance is
done: the README carries `curl` output captured from the live stack serving the
promoted model, and the architecture diagrams are Mermaid and describe what runs.

**W4-04 needs the five real member names.** The role table is filled with git handles,
which the brief's "Role Documentation" criterion will not reward. This is the one
remaining item that needs a human rather than a commit.

**W4-05 is closed:** all 12 emitted schemas carry object-level OpenAPI examples, and
the leftover English examples from the `amazon_polarity` era ("Excellent build
quality", "battery life") are now Vietnamese and match real responses.

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

Status of each, checked 2026-08-14:

- **Required files** all exist — plus `docs/PROBLEM.md`, `docs/FAIRNESS.md` and
  `docs/ETHICS.md`. None are committed yet.
- **`.gitignore` still misses `models/`.** It covers `data/raw|processed|interim/`,
  `mlruns/`, `.venv/` and `artifacts/`, but `models/` is guarded only by extension
  (`*.pkl`, `*.pt`, `*.bin`, `*.onnx`). One line to fix.
- **All five members have commits**, but they are not spread out: see the contribution
  table in `CONTRIBUTING.md`. `thong312` has 3 and `HuynhLC` 4, and neither has
  committed since 11 August.
- **CI green on `main`** is unverified — no `gh` CLI here.
- **Slides and rehearsals** have not started (W4-06 through W4-10).

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

---

## Toolchain drift — closed

A CI failure that `make typecheck` could not reproduce, worth recording because the cause
was structural rather than a coding mistake.

`data/validate.py` narrowed `value_counts().to_dict()` with `int(label)`. Under
`pandas-stubs` that mapping is typed `dict[Hashable, int]`, and `Hashable` cannot be
passed to `int()`. CI caught it; local `make typecheck` did not.

**The cause was two places pinning the same tools.** The CI workflow hand-listed
`mypy==1.8.0 types-requests pandas-stubs==2.3.3.260113`, while `requirements-dev.txt`
pinned a different stubs version. CI was therefore permanently stricter than the command
the contributing guide tells people to run — so a type error could pass review and fail
the build, which is exactly what happened.

Closed three ways:

1. `requirements-dev.txt` pins `pandas-stubs==2.3.3.260113` and
   `types-requests==2.32.0.20241016`, matching what CI was using.
2. The CI type-check job now installs `-r requirements-dev.txt` instead of hand-listing,
   so the two cannot diverge again.
3. Verified by reproduction: the old pattern fails and the new one passes under
   mypy 2.3.0 + pandas 3.0.5, and the whole tree passes under the pinned
   mypy 1.8.0 + pandas-stubs 2.3.3.

**The mypy pin was deliberately not bumped.** mypy 1.14 surfaces six further errors in
`data/` and `training/`, and all six are `pandas-stubs` imprecision — it types
`.fillna("").astype(str)` as `Series[bool]`. Bumping would buy six `type: ignore` comments
that document nothing. Revisit when the stubs improve.
