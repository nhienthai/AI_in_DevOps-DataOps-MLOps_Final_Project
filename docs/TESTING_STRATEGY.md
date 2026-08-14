# Testing strategy

Four kinds of test, each answering a different question. The rubric asks for all
four; more usefully, each catches a class of failure the others cannot.

| Type | Directory | Question it answers | Count |
|---|---|---|---|
| Unit | `tests/unit/` | Does this function do what its docstring says, in isolation? | 35 |
| Data quality | `tests/data/` | Would bad data get through the gate and into training? | 8 |
| Model | `tests/model/` | Does the model satisfy the contract the API depends on? | 10 |
| Integration | `tests/integration/` | Does the assembled HTTP surface behave, end to end? | 27 |

80 tests in total.

## Running them

```bash
make test    # unit + data + model + in-process integration, with the coverage gate
make smoke   # only the tests that need a live Docker Compose stack
```

`make test` runs `pytest -m "not slow and not integration"`. `make smoke` runs
`pytest tests/integration -m integration --no-cov` and requires
`docker compose up -d` first.

Two markers are declared in `pyproject.toml` under `--strict-markers`, so a typo in a
marker name fails the run rather than silently selecting nothing:

- `integration` — needs a live stack. Applied module-wide in
  `tests/integration/test_stack_smoke.py`.
- `slow` — loads or trains a large model. **Declared but not yet used by any test.**
  The nightly workflow that would run these is W2-11, still open.

## Unit tests

Pure functions and single objects, no network and no containers.

- `test_config.py` — settings parse from `SENTIMENT_`-prefixed variables; a blank
  reload token disables the reload endpoint rather than setting an empty secret; the
  label map must stay consistent with the three-class serving contract.
- `test_predictor.py` — `StubPredictor` is deterministic (the same text always yields
  the same label), scores stay in `[0, 1]`, confidence in `[0.5, 1]`, and
  over-long input sets `truncated`.
- `test_metrics.py` — PSI is zero against an identical distribution, rises as the
  distribution shifts, and the collectors carry the exact `http_*` / `ml_*` names the
  dashboards and alert rules query by.
- `test_runtime.py` — the bounded inference pool: a full pool rejects rather than
  queues forever, a timeout releases its slot, and a failed reload keeps the previous
  model serving.
- `test_ingest.py`, `test_preprocess.py` — column normalisation, and that splits are
  reproducible from the seed.

## Data quality tests

`tests/data/test_validate.py` drives the gate that `data/validate.py` enforces. Each
test feeds it a frame that is broken in exactly one way and asserts the run fails:
missing columns, empty text above the tolerated ratio, duplicates above the tolerated
ratio, label balance outside tolerance, too few rows. One test asserts a clean frame
passes, so the gate cannot be trivially satisfied by rejecting everything.

The point is not that the assertions hold today; it is that a silent data regression
becomes a red build instead of a quietly worse model.

## Model tests

`tests/model/` tests the *contract*, not the accuracy — accuracy belongs to the
evaluation gate in `scripts/validate_model.py`.

- Label metadata: a checkpoint whose `id2label` contradicts the serving contract is
  refused at load time rather than silently relabelling every prediction. Generic
  Hugging Face labels (`LABEL_0`…) are accepted; a contradictory mapping raises.
- Baseline predictor: trains on a handful of Vietnamese sentences and returns a label
  inside the allowed set with confidence in range.
- Latency budget: `check_latency_budget` measures p95 and compares it to the target.
- Registry: promotion transitions a run to `Production`; loading with no `Production`
  version raises rather than falling back to an arbitrary model.

## Integration tests

23 of these run in-process through FastAPI's `TestClient`, which executes the real
application lifespan — no container needed, so they run on every push:

- `test_app_health.py` — `/health` stays 200 even when model loading failed, while
  `/ready` returns `503 model_not_ready`; `/metrics` exposes both metric families.
- `test_predict.py` — the full request contract: valid prediction shape, empty text →
  422, over-long text → 413, over-sized batch → 413, a batch mixing valid and invalid
  items returns per-item results in order, and Unicode and emoji survive the round
  trip.
- `test_explain.py` — `/api/v1/explain` answers `503 explainer_not_available` while no
  explainer is installed, and the response schema is what the endpoint will return
  once one is.

The remaining 4, in `test_stack_smoke.py`, need the real stack: they assert the API is
reachable over the published port, that Prometheus has scraped it
(`up{job="sentiment-api"} == 1`), and that alert rules loaded. CI runs them in the
`container` job after building the image.

## Coverage

The gate is `fail_under = 80` in `pyproject.toml`, on from the first commit so it
cannot silently rot.

**Read the scope before quoting the number.** `[tool.coverage.run] omit` currently
excludes `src/sentiment/data/*`, `src/sentiment/models/*`, `src/sentiment/training/*`
and `src/sentiment/responsible/*`. Coverage therefore measures the serving layer plus
`config.py` — not the whole package. Tests for the excluded modules do exist and do
run; their lines are simply not counted.

That is defensible for the training code, which needs a GPU and a real dataset to
exercise honestly, and not defensible for `data/` and `models/`, which are already
tested. Narrowing the omit list to `training/*` is tracked as part of W3-12.

## What is not tested yet

- No fairness test, because `responsible/fairness.py` is a stub. The design makes
  fairness a *gate* rather than a report — a promotion rule (W2-06), a failing test
  (W2-09) and an alert (W3-06) — and none of those three exist yet.
- No transformer behaviour test on a real checkpoint (known-positive and
  known-negative cases, calibration bounds). This is what the `slow` marker and the
  nightly workflow are for.
- No load test, so the alert thresholds in `prometheus/alerts/` have not been observed
  firing under traffic (W3-04).

## See also

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — what the tests are testing
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — the PR checklist these feed
- [`docs/user-guide.md`](user-guide.md) — alert runbooks
