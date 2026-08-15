# Presentation plan — 18 minutes + 10 Q&A

Build the deck from this. Each slide lists its speaker, its time box, and the one thing
it has to land. Everything on a slide is a fact from the repo, not a claim.

**Timing budget.** The per-slide times below sum to exactly **18:00**, leaving 2 minutes
of slack inside the 15–20 window. Add them up again if you change any of them — an
earlier draft of this table summed to 20:30 and would have overrun the cap.

Rehearse to 18. If you overrun, the demo is what gets squeezed, and the demo is 15% of
the presentation mark — so cut slide 4 or 11 instead.

**Rule for every slide:** no bullet without a number or a file path behind it.

**The deck is built.** [`docs/deck.html`](deck.html) — open it in any browser, no build
step and no network. `←` `→` to move, `G` for an overview of all sixteen slides with
speaker and time, `N` to toggle speaker notes. The rail shows the current speaker and the
running total against the 18:00 budget, both derived from the slides themselves, so they
cannot drift from this table.

| # | Slide | Speaker | Min |
|---|---|---|---|
| 1 | Title & the problem | Nhiên | 0:45 |
| 2 | Roadmap — where the time goes | Nhiên | 0:30 |
| 3 | The 4% constraint | Thông | 1:00 |
| 4 | System at a glance | Sơn | 0:30 |
| 5 | Architecture end to end | Sơn | 1:15 |
| 6 | Gate one: the data | Thông | 1:00 |
| 7 | Technology trade-offs | Sơn | 0:45 |
| 8 | Ten trials, an 11-point spread | Duy | 1:00 |
| 9 | Gate two, and the bug it caught | Duy | 1:15 |
| 10 | Serving: shed, don't queue | Sơn | 1:15 |
| 11 | Atomic reload | Sơn | 0:45 |
| 12 | Five kinds of test | Huỳnh | 0:45 |
| 13 | Drift, and the reference | Huỳnh | 1:00 |
| 14 | CI/CD | Huỳnh | 0:45 |
| 15 | Fairness: measured, mitigated, gated | Nhiên | 1:45 |
| 16 | Explainability & refusal | Nhiên | 0:45 |
| 17 | **Live demo** | Sơn + Huỳnh | 1:45 |
| 18 | What we got wrong | cả nhóm | 0:45 |
| 19 | Close | Nhiên | 0:30 |

---

## 1. Title — Nhiên (0:30)

**sentiment-service** — production sentiment analysis for Vietnamese student feedback.
Team, course, date.

> "A faculty collects thirty thousand free-text comments a semester. Nobody reads them.
> We built the system that does."

## 2. Why keyword counting fails — Nhiên (1:00)

Land the problem, not the solution.

- 30,000 comments per semester, two-week deadline, Vietnamese free text
- Skimming means the loudest complaint sets the agenda regardless of frequency
- Keyword counting cannot separate `thầy dạy hay` from `thầy dạy **không** hay`
- **The cost is not reading time — it is discovering a failing course a semester late**

## 3. Requirements & success metrics — Nhiên (1:00)

Three levels, with targets. Source: `docs/PROBLEM.md`.

| Level | Metric | Target |
|---|---|---|
| Business | One semester classified | < 30 min |
| System | p95 latency | < 200 ms |
| Model | macro-F1 **and** accuracy | ≥ 0.70 / ≥ 0.85 |
| Model | Identity-pair fairness gap | ≤ 0.10 |

Say out loud why both model metrics are required: accuracy alone is satisfiable by
ignoring the minority class.

## 4. Architecture — Sơn (1:15)

The Mermaid diagram from `ARCHITECTURE.md`. Six services: `api`, `postgres`, `mlflow`,
`prometheus`, `alertmanager`, `grafana`.

Point at the two diamond gates and say: **neither is advisory.** `data/validate.py`
raises; `validate_model.py` exits non-zero.

One design decision to justify: the walking skeleton. All six services were integrated in
week 1 against a stub model, so swapping in a real model later was a config change, not a
rewrite. *This is the answer to "how did you avoid integration hell".*

## 5. Data pipeline & the quality gate — Thông (1:00)

`HF UIT-VSFC → normalise → quality gate → splits + drift reference → fingerprint`

The gate fails the run on: missing columns, empty-text ratio, duplicate ratio, a missing
class, or the rarest class below its floor.

Each split is fingerprinted with SHA-256 over its content **in order**, logged as an
MLflow param. Two runs that disagree on metrics can be checked for whether they saw the
same data.

## 6. The 4% problem — Thông (1:00)

The most important slide about the data.

| | negative | neutral | positive |
|---|---|---|---|
| train rows | 5,325 | **458** | 5,643 |

- `neutral` is **4%** of the corpus
- Accuracy 0.8629 against macro-F1 0.7149 — **the 15-point gap is this one class**
- It is why the balance rule is a per-class floor, not a deviation from 0.5
- It is why the original macro-F1 ≥ 0.92 target was wrong and was replaced

## 7. Two model families, and the experiment — Duy (1:00)

Source: `docs/EXPERIMENTS.md`. **18 MLflow runs.**

- Baseline TF-IDF + LogisticRegression: macro-F1 **0.7149**, accuracy **0.8629**
- 10 Optuna trials, one nested run each, optimising **cross-validated** macro-F1
- Trial spread **0.5975 – 0.7079** — the search did real work
- The winning trial independently chose `class_weight=balanced`

Why cross-validate inside the objective: on a single split, the trial with the luckiest
fold wins and the result does not reproduce.

**Be honest here:** the XLM-RoBERTa fine-tune is implemented and satisfies the serving
contract, but no fine-tuned checkpoint has been promoted. The baseline is what is in
`Production`.

## 8. Promotion is a gate, not a decision — Duy (1:30)

`scripts/validate_model.py` refuses to promote unless **all four** hold:

| Check | Threshold | Measured |
|---|---|---|
| macro-F1 | ≥ 0.70 | 0.7149 |
| accuracy | ≥ 0.85 | 0.8629 |
| p95 latency | < 200 ms | 0.42 ms |
| fairness gap | ≤ 0.10 | 0.0000 |

Nothing is promoted by hand. Rollback is a stage transition plus a reload — no rebuild.

**The story to tell:** the gate caught a real bug. A model trained with identity blinding
lost the flag when reloaded from disk and measured a 0.08 gap instead of 0.00. Review did
not catch it; the gate did. It is now a regression test.

## 9. Serving: what happens under load — Sơn (1:00)

The interesting part is not the happy path.

- Inference never runs on the event loop — bounded thread pool sized to
  `max_concurrent_inferences`
- Pool full → **429**, shed rather than queued forever
- Over the time budget → **504**, slot released by a done-callback
- Reload warms the replacement *before* publishing it, so a failed reload leaves the
  previous model serving. **You cannot take the service down with a bad rollback.**
- `/health` stays 200 when the model is broken; `/ready` returns 503. A model fault must
  not get the container killed while you diagnose it.
- Container: multi-stage, non-root UID 1000, base pinned by digest, `pip`/`setuptools`
  removed from the runtime layer, Trivy scan and CycloneDX SBOM in CI

## 10. Observability — Huỳnh (1:15)

- **20 collectors**, `http_*` and `ml_*`, in-process — drift and confidence are functions
  of the prediction, so exporting them elsewhere would mean shipping predictions to a
  second process
- **10 alert rules**, every one with a runbook anchor that is verified mechanically
- **3 dashboards** provisioned from files, surviving `docker compose down -v`

Justify one threshold rather than listing all ten: PSI > 0.2 is the conventional
significant-shift boundary (< 0.1 stable, 0.1–0.2 moderate), which is why the alert is
0.2 and not a number someone liked.

**Evidence it works:** `DriftDetected` was driven to `FIRING` with `scripts/load_test.py`
and confirmed received by Alertmanager. Say plainly that `HighLatencyP95` has *not* been
observed firing — the baseline answers in 0.42 ms, so nothing this stack can generate will
push p95 to 500 ms.

## 11. CI/CD — Huỳnh (0:45)

`lint → type-check → test (3.10, 3.11) → build + scan + smoke → publish`

- **159 tests**, five categories, **91% coverage** against an 85% gate
- Nightly workflow for the `slow` set and the fairness gate on a freshly trained model
- Publishes to GHCR on `main`, then pulls the published image back and checks `/ready` —
  a broken publish fails in CI, not on someone else's machine

Worth one sentence: CI once pinned mypy and the stubs separately from
`requirements-dev.txt`, so CI was stricter than `make typecheck` and caught a type error
review had passed. It now installs from the same file.

## 12. Fairness — Nhiên (2:00)

The highest-value slide. Do not rush it.

**Method.** UIT-VSFC has no demographic columns, so group-wise accuracy is impossible —
there is nobody to group by. Instead probe the model: 216 sentences, 60 identity pairs
that differ by exactly one term across gender, region and seniority. Measured **over
HTTP against the deployed service**, so the number covers the real preprocessing and
checkpoint.

| Variant | Max gap | gender | region | macro-F1 | Gate |
|---|---|---|---|---|---|
| none | 0.1417 | 0.1417 | 0.0868 | 0.7143 | fail |
| + counterfactual | 0.1080 | **0.0297** | **0.1080** | 0.7113 | fail |
| + blinding | **0.0000** | 0.0000 | 0.0000 | 0.7149 | **pass** |

**Tell the middle row as a story.** Counterfactual augmentation cut gender bias 79% and
still failed — because region became the binding constraint. Why: region terms appear
**0 times** in training data, while `thầy`/`cô` appear in 23.9%/11.1% of rows.
Augmentation cannot teach words the model never saw.

Blinding reaches exactly zero and cost **+0.0006** macro-F1 — free, which is itself the
finding: those terms carried no label signal.

**And admit the error.** The first measurement said 0.63. It was wrong: the probe used
capitalised `Thầy` and the corpus is entirely lowercase. It was measuring casing, not
bias. This is why the probe itself is unit-tested against a known injected bias.

## 13. Explainability — Nhiên (0:45)

Two methods, different questions.

- **LIME, local, live** on `POST /api/v1/explain`
- **SHAP, global, offline** — logged as an MLflow artifact per run

Show the real output for `"Giáo viên dạy rất hay nhưng tài liệu thì sơ sài"`:
`rất` +0.30, `hay` +0.26, `nhưng` −0.19, `sơ` −0.15 → **negative**. The model reads the
concession.

State the limitation: the `Predictor` protocol exposes one positive-class score, so LIME
explains the positive-class probability rather than attributing across all three classes.

## 14. Live demo — Sơn + Huỳnh (2:30)

**Rehearse this until it is boring.** Script, in order:

1. `docker compose ps` — six healthy services *(pre-warmed; do not build live)*
2. `curl /api/v1/model/info` — version 2, `Production`, real metrics, `fairness_delta: 0.0`
3. `curl /api/v1/predict` — one positive, one negative
4. `curl /api/v1/predict/batch` with an empty item — per-item error, batch survives
5. `curl /api/v1/explain` — live token attributions
6. Grafana → **Sentiment** folder → Model & Predictions, panels populated
7. `python scripts/load_test.py --scenario drift --duration 60` → Prometheus alert goes
   `pending`

**Safety rules.** Stack up and warm before you present. Have every response saved as a
fallback slide. On macOS, turn off AirPlay Receiver first — it binds port 5000 and makes
MLflow look dead. Never run `docker compose up --build` in front of the room.

## 15. What we got wrong — all (1:00)

Do not skip this. It is the difference between a demo and engineering.

- **A macro-F1 ≥ 0.92 target set for the wrong dataset.** Written for balanced binary
  English data, kept after switching to 3-class Vietnamese. Replaced with justified,
  reachable targets.
- **A fairness probe that measured capitalisation.** Fixed, then unit-tested so it cannot
  recur.
- **Coverage that measured a quarter of the package.** The omit list hid `data/`,
  `models/`, `training/` and `responsible/`. Narrowed; the real figure is 91%.
- **A train/serve skew the gate caught and review did not.**
- **Uneven contribution.** Visible in `git shortlog` and stated in `CONTRIBUTING.md`
  rather than hidden.

## 16. Close — Nhiên (0:30)

What is deployed: a gated, monitored, explainable service with a measured fairness
property. What is next: the transformer fine-tune, and the honest possibility that it does
not earn deployment on a CPU latency budget.

---

## Backup slides — only if asked

- Per-dimension fairness table and the worst remaining pairs
- Drift PSI: bin edges, why input length is the proxy, the fallback-reference failure mode
- The `score` vs three-class inconsistency and why it was left as a deliberate open item
- Compose service graph with health-check dependency order
- Ethics: uses this system refuses, and why per-lecturer scoring is one of them
