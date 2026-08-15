# Testing strategy

Four kinds of test, each answering a different question. The rubric asks for all
four; more usefully, each catches a class of failure the others cannot.

| Type | Directory | Question it answers | Count |
|---|---|---|---|
| Unit | `tests/unit/` | Does this function do what its docstring says, in isolation? | 47 |
| Data quality | `tests/data/` | Would bad data get through the gate and into training? | 21 |
| Model | `tests/model/` | Does the model satisfy its contract, and is it any good? | 30 |
| Responsible AI | `tests/responsible/` | Do the fairness and explainability instruments measure what they claim? | 32 |
| Integration | `tests/integration/` | Does the assembled HTTP surface behave, end to end? | 29 |

**159 tests**: 147 run on every push, 8 are `slow`, and 4 need a live stack.

## Running them

```bash
make test     # 147 tests, with the coverage gate
make smoke    # the 4 that need a live Docker Compose stack
pytest -m slow  # the 8 that download the dataset or a checkpoint
```

`make test` runs `pytest -m "not slow and not integration"`. `make smoke` runs
`pytest tests/integration -m integration --no-cov` and requires
`docker compose up -d` first. The nightly workflow runs the `slow` set.

Both markers are declared in `pyproject.toml` under `--strict-markers`, so a typo in a
marker name fails the run rather than silently selecting nothing:

- `integration` — needs a live stack. Applied module-wide in
  `tests/integration/test_stack_smoke.py`.
- `slow` — downloads the corpus or a checkpoint. Applied module-wide in
  `tests/model/test_model_slow.py`, and run by
  [`.github/workflows/nightly.yml`](../.github/workflows/nightly.yml).

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

**91%**, against a `fail_under = 85` gate that has been on since the first commit so it
cannot silently rot.

**Read the scope, not just the number.** `[tool.coverage.run] omit` excludes exactly two
modules, and only because they cannot be exercised honestly in CI:

- `models/transformer.py` — needs a real checkpoint and a GPU.
- `training/train.py` — needs the full corpus and a live MLflow server.

Everything else is measured: `data/`, `models/baseline.py`, `models/registry.py`,
`responsible/`, `training/evaluate.py`, `training/tune.py` and all of `serving/`.

This was not always true. The omit list previously also excluded `data/*`, `models/*`,
`training/*` and `responsible/*`, so the reported figure described the serving layer
alone — a real number measured over a quarter of the package. It was narrowed as part of
W3-12, which is why the count of measured statements roughly quadrupled while the
percentage barely moved.

The two omitted modules are still covered by behaviour: the `slow` set trains a real
model through `train.py`'s helpers and loads a real transformer checkpoint. Those lines
are exercised nightly; they are simply not counted on every push.

## Responsible AI tests

`tests/responsible/` tests the *instruments*, which is easy to skip and the reason
fairness work is often unfalsifiable. A probe that always returns zero would look like a
fair model.

- **The probe detects known bias.** A deliberately gendered scorer must produce a gap on
  the `gender` dimension and **zero** on the others; a constant scorer must produce zero
  everywhere. Both are asserted.
- **Only same-dimension groups are compared** — a gender term against a seniority term
  would be a meaningless pair.
- **Blinding makes an identity pair byte-identical**, which is why the delta is exactly
  zero rather than approximately zero.
- **Longest-term-first substitution**, so replacing `thầy` cannot corrupt
  `thầy người hà nội`.
- **LIME is deterministic** for the same input, and its label agrees with the predictor's.
- **SHAP shape normalisation**, including the three-dimensional multiclass array that
  silently reduces along the wrong axis if handled naively.
- **The report's arithmetic**, including that it describes the *best* variant rather than
  the last, and that a mitigation which failed is still reported.

## What is not tested yet

- **No transformer behaviour on a fine-tuned checkpoint.** The `slow` set loads
  `xlm-roberta-base` and asserts the serving contract, but its head is untrained, so
  accuracy claims about the transformer are untested because no such model exists yet.
- **`HighLatencyP95` has never been observed firing.** The baseline answers in 0.42 ms,
  so `load_test.py --scenario burst` cannot push p95 past the 500 ms threshold. The rule
  is verified by inspection only; `DriftDetected` was verified for real.
- **No test asserts the drift reference travels with the model.** Serving falls back to a
  bootstrap reference when the artifact is missing, and nothing fails if it does — which
  is exactly how a permanently drifting signal goes unnoticed.

## See also

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — what the tests are testing
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — the PR checklist these feed
- [`docs/user-guide.md`](user-guide.md) — alert runbooks
