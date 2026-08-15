# Contributing

Team roles and the working agreement for the DDM501 final project.

## Team roles

Ownership is by directory and is disjoint on purpose. It keeps merge conflicts rare,
and it makes the git history evidence of who did what without anyone having to argue
for it during the individual contribution assessment.

| | Member | Git identity | Area | Owns |
|---|---|---|---|---|
| M1 | **Lý Minh Thông** | `thong312` | Data & Features | `src/sentiment/data/`, `tests/data/` |
| M2 | **Dương Thành Duy** | `dtduy77` | Training & Experiments | `src/sentiment/models/`, `src/sentiment/training/`, `scripts/train_model.py`, `scripts/validate_model.py`, `scripts/evaluate_model.py`, `notebooks/`, `tests/model/` |
| M3 | **Bùi Vân Sơn** | `sontv6666` | Serving & Containers | `src/sentiment/serving/`, `Dockerfile`, `docker-compose.yml` |
| M4 | **Lê Công Huỳnh** | `HuynhLC` | Monitoring & CI/CD | `prometheus/`, `grafana/`, `alertmanager/`, `scripts/load_test.py`, `.github/workflows/`, `tests/integration/` |
| M5 | **Thái Bình Nhiên** | `Nhien Thai` | Responsible AI & Docs | `src/sentiment/responsible/`, `scripts/run_fairness_probe.py`, `docs/`, `README.md`, `ARCHITECTURE.md` |

The git identity column exists so that any commit can be traced to a person without
guessing. Each mapping is unambiguous from the commit email — `lyminhthong312@`,
`duythduong.2003@`, `sontv6666@`, `huynhlc1281@`, `thaibinhnhien@`.

## What each member built

Not a list of directories — the substance, so that each person can be asked about their
own work and answer it.

**M1 — Lý Minh Thông, Data & Features.** The ingestion and normalisation layer, and the
quality gate that *fails the run* rather than warning: schema, empty-text and duplicate
ratios, and a floor on the rarest class. Deterministic stratified splits, and the drift
reference distribution that travels with the model so the PSI metric compares against
the data that model was actually trained on.

**M2 — Dương Thành Duy, Training & Experiments.** Both model families: the TF-IDF +
LogisticRegression baseline and the XLM-RoBERTa fine-tune, with balanced class weights
for the 4% `neutral` class, early stopping, a cosine schedule and a multi-GPU fix for
Kaggle. The MLflow registry client, the Optuna sweep with one nested run per trial, and
the promotion gate.

**M3 — Bùi Vân Sơn, Serving & Containers.** The whole HTTP surface and its contract:
typed errors, per-item batch semantics, and an inference runtime that sheds load rather
than queueing without limit, times out rather than hanging, and swaps models atomically
so a failed reload leaves the previous model serving. The multi-stage non-root image
with a pinned base digest and build tools stripped from the runtime layer.

**M4 — Lê Công Huỳnh, Monitoring & CI/CD.** The metric vocabulary the rest of the system
is observed through — 20 collectors across `http_*` and `ml_*`, including PSI drift — the
six-service Compose stack with health-check ordering, ten alert rules with thresholds
that are argued for rather than guessed, three provisioned Grafana dashboards, the load
generator, and the CI pipeline from lint through image scanning to smoke test.

**M5 — Thái Bình Nhiên, Responsible AI & Docs.** The fairness probe and the two
mitigations, with the before/after measurement that showed one of them failing and why;
SHAP global importance and LIME local explanations; the ethics and privacy analysis; and
the documentation set, including the problem definition and success metrics.

## Contribution by commit

From `git shortlog -sne origin/main`:

| Member | Commits | Lines added |
|---|---|---|
| Bùi Vân Sơn | 17 | +2,988 |
| Thái Bình Nhiên | 16 | +4,083 |
| Dương Thành Duy | 10 | +1,102 |
| Lê Công Huỳnh | 4 | +698 |
| Lý Minh Thông | 3 | +396 |

Read these honestly rather than as a ranking. Line counts favour whoever wrote prose and
moved directories, and commit counts favour whoever integrated. The distribution is
uneven: M1 and M4 have the fewest commits and neither has committed since 11 August,
while M3 and M5 carry most of the history. Anyone reviewing individual contribution
should ask each member about the substance above, not about the size of the diff.

## Branching

```
<initials>/<short-description>
```

Initials per member: `lmt` (Thông), `dtd` (Duy), `bvs` (Sơn), `lch` (Huỳnh),
`tbn` (Nhiên). Existing branches use the handle instead — `sontv/m3-serving`,
`huynhlc/monitoring-cicd` — which is equally traceable; the point is that a branch names
its owner.
Branch from `main`; never commit to `main` directly.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), with these types:

| Type | Use for |
|---|---|
| `feat:` | new behaviour |
| `fix:` | a defect in existing behaviour |
| `test:` | tests only |
| `refactor:` | behaviour-preserving restructuring |
| `docs:` | documentation |
| `chore:` | tooling, dependencies, housekeeping |
| `ci:` | pipeline changes |

An optional scope narrows it: `feat(train): add early stopping`.

Write the subject as an instruction — "add the drift reference", not "added" or "adds"
— and explain *why* in the body when the reason is not obvious from the diff.

## Pull requests

Every PR must satisfy all of these before review:

- [ ] `make test` passes and the coverage gate holds
- [ ] `make lint` clean (flake8 + black + isort)
- [ ] `make typecheck` clean (mypy)
- [ ] CI green on the branch
- [ ] Docs updated when behaviour or configuration changed
- [ ] `TASKS.md` status updated for the tasks the PR touches

Two rules that are not negotiable:

1. **Nobody merges their own PR.**
2. **Every member reviews at least one PR per week outside their own area.** This is
   what stops five people from each knowing only one fifth of the system — and the
   presentation includes individual questions about the whole thing, not just your
   directory.

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
make install          # editable install + dev dependencies
make test
```

Run the full stack with `make up`, tear it down with `make down` (which passes `-v`,
discarding volumes — required after changing anything under `prometheus/` or
`grafana/provisioning/`).

Configuration is read from `SENTIMENT_`-prefixed environment variables, or a `.env`
file copied from `.env.example`. Never commit `.env`.

## Definition of done

A task is done when it is merged to `main` with CI green — not when the code works
locally. `TASKS.md` uses `TODO` → `WIP` → `REVIEW` (PR open) → `DONE` (merged, CI
green), plus `BLOCKED` for work waiting on something outside the task — a GPU, a
decision, another member. The status column is only meaningful if people update it in
the same PR as the work, and marking a row `DONE` in bulk without checking it is how a
board stops being trustworthy.

## See also

- [`README.md`](README.md) — quickstart and API examples
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design and trade-offs
- [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md) — what to test and where
- [`TASKS.md`](TASKS.md) — the task board
