# Q&A preparation

Three likely questions per member, on their own area, with an answer each. The brief says
individual grades may be adjusted from individual Q&A, so each member owns their section
and should be able to answer without the deck.

**How to answer well here:** name the file, give the number, and say the trade-off. "We
handled that" is a weak answer; "`validate.py` raises `DataQualityError` and there is a
test that feeds it a frame with 0.5% neutral to prove it fails" is a strong one.

**If you do not know:** say so and say where you would look. Inventing a number in front
of someone who can open the repo is the worst outcome available.

---

## M1 — Lý Minh Thông, Data & Features

**Q1. Your quality gate rejects a split when the rarest class falls below 2%. Why 2%, and
why a floor rather than checking the distribution matches the training prior?**

Because a prior only exists for a balanced problem. The gate originally checked that the
positive ratio stayed within 0.1 of 0.5 — correct for binary `amazon_polarity`, and
guaranteed to fail every healthy run on UIT-VSFC, where `neutral` is 4%. What actually
needs catching is a class *disappearing*: a split with almost no `neutral` rows cannot
train or score that class, and the macro-F1 it produces is meaningless. 2% sits below the
observed 4.0% train / 4.6% validation / 5.3% test so it does not fire on healthy data, and
far enough above zero to catch a broken split. The gate also asserts all three classes are
present, which catches the degenerate case the ratio alone would miss.

**Q2. What stops a model being trained on data that never passed the gate?**

Training enters through one function, `load_validated_splits` in `training/train.py`,
which normalises, calls `validate()`, and fingerprints each split. For a while this was
not true — `train.py` called `load_dataset()` directly and the gate guarded data nobody
trained on. That was W2-04 and it is closed. There is also a `Settings` validator that
raises if `dataset_name` and `model_dataset_name` ever disagree, so the two-datasets-at-
once state cannot come back.

**Q3. You fingerprint the data with SHA-256. Is that not what DVC is for, and what does
the hash actually protect against?**

DVC solves versioning for data that *changes* and needs remote storage. UIT-VSFC is
immutable on the HuggingFace hub, so a remote would add infrastructure for no benefit.
What we needed was the ability to prove which rows produced a model. The hash covers the
text and labels **in order**, with delimiters, so reordering changes it and two datasets
that concatenate to the same bytes do not collide — there is a test for each of those.
It goes to MLflow as a param, so two runs that disagree on metrics can be checked for
whether they even saw the same data.

---

## M2 — Dương Thành Duy, Training & Experiments

**Q1. Your Optuna objective is cross-validated rather than a single held-out score. That
is 3× the compute. Justify it.**

With `neutral` at 458 rows, a single split can move macro-F1 on that class by several
points. On a single split the trial that draws the favourable fold wins, and the sweep
then reports a configuration that does not reproduce. The trial spread was 0.5975 to
0.7079 — an 11-point range — so the ranking is load-bearing, and it has to be a ranking of
configurations rather than of fold luck. The standard deviation (±0.004) is what makes
the later mitigation comparison believable: a 0.003 difference between variants would be
noise without it.

**Q2. The transformer is not deployed. Was the work wasted, and how would you decide
whether to ship it?**

The implementation is complete and tested — the `slow` suite loads a real checkpoint and
asserts it satisfies the serving contract — but no fine-tuned model has been promoted, so
the baseline is what is in `Production`. Two conditions decide it: macro-F1 ≥ 0.80 against
the baseline's 0.7149, and p95 < 200 ms on CPU. `xlm-roberta-base` is roughly four times
DistilBERT's size and the 200 ms budget was set for DistilBERT, so the second is the real
risk. If it beats the baseline on quality and misses on latency, the honest outcome is to
keep the baseline and report the transformer as an experiment that did not earn
deployment. The registry makes that a stage transition, not a rewrite.

**Q3. Your promotion gate has four checks. What happens when a model passes them and is
still wrong?**

The gate bounds what we can measure before deployment; it is not a correctness proof. That
is what the serving-side alerts are for: `FairnessRegression` fires if a deployed model
exceeds the same 0.10 threshold the gate applies, `DriftDetected` fires when inputs move
away from the training reference, and `PredictionSkew` catches the output distribution
collapsing. We have already had one case of exactly this shape — a model that passed
training-time fairness measured 0.08 after a joblib round-trip because the blinding flag
did not persist. The gate caught it because it re-measures the *loaded* artifact rather
than trusting the training-time number.

---

## M3 — Bùi Vân Sơn, Serving & Containers

**Q1. Why 429 when the inference pool is full, instead of queueing the request?**

Because an unbounded queue converts an overload into a timeout for everybody. The pool is
sized to `max_concurrent_inferences`; a request that cannot get a slot within
`queue_timeout_seconds` is rejected with `429 inference_overloaded` and the client can
retry or back off. The alternative — accept everything and let latency grow — makes p95
meaningless and eventually exhausts memory. It is visible too: `ml_inference_overloads_total`
and `ml_inference_queue_depth` are exported, so shedding shows up on the dashboard rather
than looking like a mystery slowdown.

**Q2. `/health` returns 200 when the model failed to load. Is that not lying about
health?**

They answer different questions, deliberately. `/health` is liveness: the process is
running and can serve HTTP. `/ready` is readiness: a warmed model is loaded and
predictions will succeed. If liveness went red on a model fault, the orchestrator would
kill and restart the container in a loop while you were trying to diagnose it, and you
would lose the logs each time. So `/health` stays 200, `/ready` returns
`503 model_not_ready`, the Docker `HEALTHCHECK` targets `/ready` so a broken release is
not marked healthy, and `ModelNotLoaded` alerts on the metric.

**Q3. Walk me through what happens if a model reload fails in production.**

`InferenceRuntime.reload` builds the replacement and runs a warm-up prediction through
`validate_predictions` *before* publishing it. If any of that raises, the exception is
caught, `ml_model_load_failures_total` increments, the error is stored for
`/model/info`, and the previous predictor is still the one bound — so the service keeps
serving the old model and reports the failure. Publishing is a single attribute
assignment under a lock, so there is no window where a half-loaded model answers requests.
The practical consequence is that a bad rollback cannot take the service down, which is
what makes rollback safe enough to do under pressure.

---

## M4 — Lê Công Huỳnh, Monitoring & CI/CD

**Q1. Where do your alert thresholds come from? Pick one and justify it.**

`DriftDetected` fires at PSI > 0.2 because that is the conventional interpretation
boundary — below 0.1 stable, 0.1 to 0.2 moderate, above 0.2 significant. Choosing a
measure with an established reading is why the threshold is defensible rather than a
number someone liked. `HighLatencyP95` is at 500 ms, deliberately 2.5× the 200 ms SLO, so
ordinary jitter does not page anyone. `HighErrorRate` counts only 5xx: 422 and 413 are
client mistakes, and an alert that fires because someone sent bad input trains people to
ignore alerts.

**Q2. Have you actually seen these alerts fire, or do you just trust the expressions?**

`DriftDetected` was driven to `FIRING` for real with `scripts/load_test.py --scenario
drift` and confirmed received by Alertmanager with `severity=warning, state=active`.
`HighLatencyP95` has **not** been observed firing, and I will not claim otherwise: the
baseline answers in 0.42 ms, so no concurrency this stack can generate pushes p95 to
500 ms. It is verified by inspection only, and it becomes testable when the transformer
lands. The rest were verified as loaded and parsing — all ten rules, across three groups.

**Q3. Your dashboards are provisioned from files. Why does that matter, and what breaks
if someone edits one in the Grafana UI?**

Provisioning from files means the dashboards survive `docker compose down -v` and arrive
with the stack on a machine that has never run it — which is what makes the demo
reproducible. A UI edit is not persisted back to the file, so it is lost on the next
provisioning pass; that is the intended trade. The related trap is that Grafana caches
provisioning in its volume, so editing a datasource file and restarting appears to do
nothing — you need `down -v`. That is written into the runbook because it costs an hour
the first time.

---

## M5 — Thái Bình Nhiên, Responsible AI & Docs

**Q1. Your fairness gap is exactly 0.0000. That looks too good. Convince me it is real.**

It is exact by construction rather than by training, which is why it is exactly zero and
not approximately zero: identity blinding replaces identity terms with a neutral
placeholder before vectorising, so the two halves of an identity pair become the *same
string*. There is a test asserting exactly that. The number is also measured twice —
in-process at promotion time, and over HTTP against the deployed service, which is the
0.0000 on the dashboard. And I would immediately add the caveat: blinding guarantees
parity only on the terms in its list. It does nothing for a term nobody thought of, or for
bias expressed through correlated wording rather than an explicit identity word.

**Q2. Counterfactual augmentation is the standard mitigation and it failed for you. Why
should I believe your measurement rather than your implementation?**

Because the failure is explained by a number rather than a shrug. It cut the *gender* gap
79%, from 0.1417 to 0.0297 — so the mechanism works where it has material. It did not move
*region*, and region then became the binding constraint at 0.1080. The reason is that
region terms like `thầy người hà nội` appear **0 times** in the training corpus, while
`thầy` and `cô` appear in 23.9% and 11.1% of rows. Augmentation can only teach the model
about words it has seen. I reported the failure rather than dropping it, because a
negative result about a method is still a result — and the per-dimension breakdown is what
makes it diagnosable instead of just disappointing.

**Q3. What is the biggest weakness in your Responsible AI work?**

That the probe is synthetic and its coverage is a choice I made. The templates isolate the
identity term, which is what makes the comparison clean, but no student wrote those
sentences, and a gap the probe does not measure is not evidence that no gap exists. The
concrete failure I already hit is instructive: the first measurement reported 0.63 because
the probe used capitalised `Thầy` while the corpus is entirely lowercase — it was measuring
casing, not bias. That is now unit-tested against a known injected bias, so the instrument
is checked rather than trusted. The second weakness is that parity says nothing about
accuracy, which is why the gate requires macro-F1 alongside it.

---

## Questions any of us might get

**"Why three classes and Vietnamese, when the spec said binary English?"**
The dataset changed to UIT-VSFC, and the brief names multi-language support as a challenge
for this topic — so XLM-RoBERTa turns it into a property of the design rather than future
work. The mistake was leaving the two configurations coexisting for a while; that is
closed, with a config validator that makes it impossible to reintroduce.

**"Your macro-F1 is 0.71. Is that good?"**
It is honest. Accuracy is 0.8629; the 15-point gap is the 4% `neutral` class, which
macro-F1 weights equally with the other two. The original 0.92 target was written for
balanced binary data and would have meant nearly solving the minority class. We replaced
it with targets that are justified and reachable, and required accuracy alongside macro-F1
so the target cannot be met by ignoring `neutral`.

**"What would you do with another two weeks?"**
Fine-tune and honestly evaluate the transformer, including whether it earns its latency.
Then resolve the one deliberate inconsistency we left: `score` is the positive-class
probability while labels are three-class, so a `neutral` prediction reports
`probs[positive]`. It is a breaking API change and we chose to do it deliberately rather
than quietly.

**"Contribution looks uneven — explain."**
It is, and it is in `CONTRIBUTING.md` rather than hidden. Ask each of us about the
substance of our own area; that is what the section per member is for.
