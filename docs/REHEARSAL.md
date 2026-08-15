# Rehearsal checklist

W4-07 and W4-09 must run on hardware that has never built this project. Docker layer
caches and leftover volumes hide broken configuration, and the grader's machine will not
have them.

A dry run has already been done from a fresh `git clone` — findings are at the bottom, and
three of them will bite you if you skip this.

## Before you start

- [ ] A machine that has **never** run this project. Not "I ran `down -v`" — a different
      machine, or at minimum `docker system prune -a` plus a clone into a new directory.
- [ ] The clone is of `main`, not of a feature branch.
- [ ] **MLflow is on `localhost:5001`, not 5000.** macOS AirPlay Receiver binds `*:5000`
      and makes MLflow answer 403, so the stack publishes 5001 instead. Demo from 5001
      and there is nothing to turn off. This already cost us one debugging session.
- [ ] Host ports free: 8000, 5001, 9090, 9093, 3000. Check with
      `lsof -nP -iTCP:8000 -sTCP:LISTEN` per port. A second monitoring stack on the same
      machine is the usual culprit.
- [ ] Stopwatch. Target is 18 minutes of content.

## Rehearsal 1 — cold, verbose, timed loosely

The point is to find what breaks, not to look good.

```bash
git clone <repo> sentiment-rehearsal && cd sentiment-rehearsal
cp .env.example .env          # required: compose refuses to start without it
docker compose up -d --build  # first build downloads CPU torch — several minutes
```

- [ ] Time the build and **write the number down.** This is what tells you whether the
      demo can be built live (it cannot) or must be pre-warmed (it must).
- [ ] All six services reach `healthy`:
      `docker compose ps --format "table {{.Service}}\t{{.Status}}"`
- [ ] `curl -s localhost:8000/ready` → `{"status":"ready",...}`
- [ ] Grafana at `localhost:3000` shows the **Sentiment** folder with three dashboards
- [ ] Prometheus at `localhost:9090` → Status → Rules shows **10** rules
- [ ] `curl -s localhost:5001/health` → 200 *(403 with `Server: AirTunes` means you are
      on 5000 — MLflow is published on 5001)*
- [ ] Every step of the demo script in [`PRESENTATION.md`](PRESENTATION.md) §14, in order
- [ ] Every member speaks their own slides at least once

Record every surprise in a list. Do not fix anything during the run.

## Between rehearsals — W4-08

- [ ] Fix everything rehearsal 1 surfaced
- [ ] Re-run `make lint`, `make typecheck`, `make test` after the fixes
- [ ] If a fix touched compose, provisioning or the Dockerfile, the next rehearsal must
      start from `down -v` again — provisioning is cached in Grafana's volume

## Rehearsal 2 — W4-09, as if graded

- [ ] Clean clone again, from scratch
- [ ] **Stack pre-warmed before the clock starts.** Never build in front of the room.
- [ ] Timed to 18 minutes, hard stop
- [ ] Every member speaks; nobody reads the slide aloud
- [ ] Demo runs from the saved script with no improvisation
- [ ] Fallback slides open in a second window, in case the network or Docker misbehaves

## Fallbacks to prepare

The demo is 15% of the presentation mark. Assume something fails.

- [ ] Screenshot of `docker compose ps` with six healthy services
- [ ] Saved JSON for `/model/info`, `/predict` (positive and negative), `/predict/batch`
      with an invalid item, and `/explain`
- [ ] Screenshot of each of the three Grafana dashboards **with data on them**
- [ ] Screenshot of Prometheus showing `DriftDetected` in `FIRING`, and of Alertmanager
      having received it
- [ ] The fairness before/after table as a slide, not a live query

## What the dry run already found

Run from a fresh clone of `main`. Three of these are worth knowing before you present.

**1. Required files are all present and nothing sensitive leaks.** `README.md`,
`ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`,
`docker-compose.yml`, `.github/workflows/` all arrive. No `data/`, `mlruns/`,
`artifacts/`, `.venv/` or `.env` is tracked.

**2. The test suite passes from a clean clone** — 147 tests, no configuration needed
beyond the dependencies.

**3. Compose refuses to start without `.env`, with a clear message.** The error names the
variable and tells you where to set it:

```
required variable GRAFANA_ADMIN_PASSWORD is missing a value: set GRAFANA_ADMIN_PASSWORD in .env
```

That is the intended behaviour — failing loudly beats booting Grafana with a blank admin
password — but it means **`cp .env.example .env` is not optional** and belongs in the very
first line of the demo script.

**4. macOS AirPlay Receiver takes port 5000.** MLflow answered `403` with
`Server: AirTunes/…` while being perfectly healthy inside the network. Checked with
`docker compose exec api python -c "import urllib.request;
print(urllib.request.urlopen('http://mlflow:5000/health').status)"` → 200. Fixed by
publishing MLflow on `127.0.0.1:5001:5000`, so no machine setting has to change.

**5. `docker compose up | tail` hides the real exit code.** A port conflict can look like
a successful start because the pipeline reports `tail`'s status. Read the output rather
than trusting the exit code — this cost us one wrong conclusion during development.

## Known-imperfect, so nobody is surprised on stage

- **`HighLatencyP95` cannot be demonstrated.** The baseline answers in 0.42 ms; no
  concurrency this stack can generate pushes p95 to 500 ms. If asked, say so — the rule is
  verified by inspection, and `DriftDetected` is the one verified firing for real.
- **`FairnessUnmeasured` will look like it should be firing.** The deployed model's gap is
  genuinely 0.0000 because of identity blinding, which is exactly the ambiguity that alert
  is `info` rather than a page. `/model/info` distinguishes the cases: `null` means
  unmeasured, `0.0` means measured as zero.
- **MLflow shows only baseline runs.** No transformer has been promoted. Do not let a
  question imply otherwise — see [`EXPERIMENTS.md`](EXPERIMENTS.md).

## See also

- [`PRESENTATION.md`](PRESENTATION.md) — slide-by-slide plan and the demo script
- [`QA_PREP.md`](QA_PREP.md) — three questions and answers per member
- [`user-guide.md`](user-guide.md) — the runbooks, if something breaks live
