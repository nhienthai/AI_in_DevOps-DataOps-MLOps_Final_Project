# Contributing

Team roles and the working agreement for the DDM501 final project.

## Team roles

Ownership is by directory and is disjoint on purpose. It keeps merge conflicts rare,
and it makes the git history evidence of who did what without anyone having to argue
for it during the individual contribution assessment.

| | Member | Area | Owns |
|---|---|---|---|
| M1 | `thong312` | Data & Features | `src/sentiment/data/`, `tests/data/` |
| M2 | `dtduy77` | Training & Experiments | `src/sentiment/models/`, `src/sentiment/training/`, `scripts/train_model.py`, `scripts/validate_model.py`, `scripts/evaluate_model.py`, `notebooks/`, `tests/model/` |
| M3 | `sontv6666` | Serving & Containers | `src/sentiment/serving/`, `Dockerfile`, `docker-compose.yml` |
| M4 | `HuynhLC` | Monitoring & CI/CD | `prometheus/`, `grafana/`, `alertmanager/`, `scripts/load_test.py`, `.github/workflows/`, `tests/integration/` |
| M5 | `Nhien Thai` | Responsible AI & Docs | `src/sentiment/responsible/`, `scripts/run_fairness_probe.py`, `docs/`, `README.md`, `ARCHITECTURE.md` |

> **Members are listed by git handle and must be replaced with real full names.**
> The submission requirements ask for documented individual roles, and the ±20%
> contribution adjustment is assessed partly from this table. Git handles are a weak
> answer to "who is this person"; fix this before submitting.

Current contribution by area, from `git shortlog` on `origin/main`:

| Member | Commits | Lines |
|---|---|---|
| `sontv6666` | 17 | +2,988 |
| `Nhien Thai` | 12 | +4,083 |
| `dtduy77` | 10 | +1,102 |
| `HuynhLC` | 4 | +698 |
| `thong312` | 3 | +396 |

Line counts favour whoever wrote prose and moved directories, so read them alongside
the commits rather than as a ranking.

## Branching

```
<initials>/<short-description>
```

For example `nt/data-quality-gate`, `sontv/m3-serving`, `huynhlc/monitoring-cicd`.
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
green), and the status column is only meaningful if people update it in the same PR as
the work.

## See also

- [`README.md`](README.md) — quickstart and API examples
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design and trade-offs
- [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md) — what to test and where
- [`TASKS.md`](TASKS.md) — the task board
