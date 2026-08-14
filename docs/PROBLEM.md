# Problem definition, requirements and success metrics

## 1. Business context

Vietnamese universities collect end-of-semester feedback from every student on every
course. A mid-sized faculty gathers tens of thousands of free-text comments per
semester, in Vietnamese, on a two-week deadline.

What happens to them today is the problem. Nobody reads thirty thousand comments, so
one of two things occurs: they are skimmed, and the loud complaints set the agenda
regardless of how common they are; or they are counted with a keyword list, which
cannot tell *"thầy dạy hay"* from *"thầy dạy không hay"*. Either way the actionable
signal — which courses are deteriorating, which complaints are systemic rather than
isolated — arrives after the next semester's timetable is already fixed.

The cost is not the reading time. It is that a department discovers a failing course a
semester late.

## 2. Problem statement

**Classify Vietnamese student course feedback into negative, neutral or positive, as a
production service that a faculty's existing tooling can call, fast enough to process a
semester's backlog in minutes and transparent enough that a human can challenge any
individual result.**

Three parts of that sentence are load-bearing:

- **Vietnamese.** English sentiment models are not a substitute. The domain is
  Vietnamese student writing, with its own honorifics, regional variation and slang.
- **As a service, not a notebook.** The output has to reach an existing system, which
  means an HTTP contract, versioning, monitoring and an operable deployment.
- **Challengeable.** A sentiment label attached to a named lecturer must come with an
  explanation, or it should not be produced at all. See
  [`ETHICS.md`](ETHICS.md) for the uses this system refuses.

## 3. Users and use cases

| User | What they need | How the system serves it |
|---|---|---|
| **Academic affairs officer** | Which courses are trending negative this semester | Batch classification of a whole semester's export via `POST /api/v1/predict/batch` |
| **Department head** | Which specific complaints are systemic | Labelled comments they can filter and read, with `confidence` to prioritise |
| **Quality assurance team** | Evidence that the numbers are trustworthy | `POST /api/v1/explain` for any disputed comment; fairness report; model version on every response |
| **Platform engineer** | To run this without becoming an ML expert | One `docker compose up`, health and readiness endpoints, ten alerts with runbooks |
| **The team itself** | To improve the model without breaking production | MLflow experiment history, a promotion gate, and rollback by stage transition |

**Primary flow.** An officer exports a semester of feedback, the faculty's tooling
posts it in batches of 64, and the service returns a label, a score and a confidence
for each. Comments below the confidence threshold are routed for human reading rather
than counted.

**Not a use case:** scoring an individual lecturer for an employment decision. Stated
here as well as in `ETHICS.md` because a requirement document that omits it invites it.

## 4. Requirements

### Functional

| ID | Requirement | Priority |
|---|---|---|
| F1 | Classify a single Vietnamese text into negative / neutral / positive with a confidence | Must |
| F2 | Classify a batch, returning per-item results where one bad item does not fail the batch | Must |
| F3 | Report which model version produced any prediction | Must |
| F4 | Expose model metadata: version, stage, metrics, fairness delta, training run | Must |
| F5 | Reject invalid input with a typed, actionable error rather than a 500 | Must |
| F6 | Return token-level explanations for a single prediction | Should |
| F7 | Swap the served model without redeploying the container | Should |
| F8 | Flag truncated input so a caller knows the verdict is partial | Should |
| F9 | Accept feedback on a prediction to drive retraining | Won't (this timeline) |

### Non-functional

| ID | Requirement | Target | Priority |
|---|---|---|---|
| N1 | Latency | p95 < 200 ms on CPU, single replica | Must |
| N2 | Throughput | ≥ 20 req/s sustained on a laptop | Must |
| N3 | Availability | Liveness independent of model state, so a model fault does not kill the container | Must |
| N4 | Bounded resources | Reject rather than queue without limit under overload | Must |
| N5 | Reproducibility | Every model traceable to a run, its params, and a dataset fingerprint | Must |
| N6 | Fairness | Worst identity-pair score delta ≤ 0.10, enforced at promotion | Must |
| N7 | Privacy | No request body persisted, logged, or recoverable from metrics | Must |
| N8 | Portability | Runs from a clean `git clone` with Docker and nothing else | Must |
| N9 | Observability | System and model metrics, with alerts that have runbooks | Must |
| N10 | Maintainability | > 80% test coverage, lint and type checks enforced in CI | Should |
| N11 | Horizontal scaling | — | Won't (drift window is in-memory; documented, not engineered) |

## 5. Success metrics

Three levels, because a model metric alone cannot tell you whether the system works.

### Business

| Metric | Target | How it is known |
|---|---|---|
| Time to classify one semester (≈30k comments) | < 30 minutes | 30,000 ÷ 20 req/s ≈ 25 min at N2 |
| Share of comments needing human reading | < 25% | Rate of predictions below the 0.7 confidence threshold |
| Feedback turnaround | Same week rather than next semester | Consequence of the two above |

### System

| Metric | Target | Where it is measured |
|---|---|---|
| p95 latency | < 200 ms | `http_request_duration_seconds`, `HighLatencyP95` at 500 ms |
| 5xx share | < 1% | `http_requests_total`, `HighErrorRate` at 5% |
| Availability | > 99% of scrape intervals up | `up{job="sentiment-api"}`, `APIDown` |
| Model load success | Failure leaves the previous model serving | `ml_model_load_failures_total`, atomic reload |

### Model

| Metric | Target | Measured |
|---|---|---|
| Baseline macro-F1 | ≥ 0.70 | **0.7149** on the held-out test split |
| Baseline accuracy | ≥ 0.85 | **0.8629** |
| Transformer macro-F1 | ≥ 0.80 | Pending GPU fine-tuning |
| Transformer accuracy | ≥ 0.92 | Pending |
| Fairness max identity-pair delta | ≤ 0.10 | **0.0000** after identity blinding |
| Input drift | PSI < 0.2 | `ml_drift_psi`, `DriftDetected` |

**Why macro-F1 and accuracy together, and why not 0.92 macro-F1.** `neutral` is 458 of
11,426 training rows — 4%. Macro-F1 weights it equally with the other two classes, so
it sits far below accuracy by construction. A macro-F1 target of 0.92 would require
very nearly solving the minority class and would fail a gate that never fitted the
data. Accuracy alone is worse: a model that ignores `neutral` entirely still scores
well on it. Requiring both is what makes the target both reachable and meaningful.

## 6. Scope and constraints

**In scope.** Three-class Vietnamese sentiment; single and batch online inference;
LIME local and SHAP global explanations; identity-pair fairness auditing with
mitigation; the full MLOps stack — tracking, registry, serving, monitoring, CI.

**Out of scope, with reasons rather than silence:**

- *Aspect-based sentiment* ("the lecturer was good but the room was cold" as two
  judgements). Needs differently annotated data.
- *Sarcasm as a solved capability.* Treated as a documented failure mode instead; see
  [`ETHICS.md`](ETHICS.md).
- *Feedback-driven continuous retraining.* Designed and rejected for this timeline: it
  roughly doubles the service count for a benefit the drift metric already delivers,
  and it is the component most likely to be half-finished at a deadline.
- *Kubernetes.* Docker Compose is what the brief requires and what one laptop needs.
- *Horizontal scaling.* The in-memory drift window would have to move to a shared
  store. Documented as a limitation rather than engineered around.

**Constraints.** Four calendar weeks. CPU-only inference on demo hardware, so model
size is a latency decision and not only an accuracy one. GPU fine-tuning limited to
Kaggle's free tier. GitHub Actions free-tier runners, which is why transformer training
cannot run on every push.

## See also

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — how these requirements are met
- [`FAIRNESS.md`](FAIRNESS.md) — N6, measured
- [`ETHICS.md`](ETHICS.md) — N7, and the uses this system refuses
- [`../TASKS.md`](../TASKS.md) — delivery status against this scope
