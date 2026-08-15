# Experiment record

What was tried, what the numbers were, and what each result changed. Every figure here
comes from a run in the `sentiment-analysis-uit-vsfc` MLflow experiment and can be opened
by run ID.

**18 runs**: 8 model runs plus 10 nested Optuna trials.

## The dataset, and why it shapes everything

`tridm/UIT-VSFC` — Vietnamese student course feedback, three classes.

| Split | Rows | negative | neutral | positive | Rarest class |
|---|---|---|---|---|---|
| train | 11,426 | 5,325 | **458** | 5,643 | **4.0%** |
| validation | 1,583 | — | — | — | 4.6% |
| test | 3,166 | — | — | — | 5.3% |

`neutral` at 4% is the single fact that explains most of what follows: the gap between
accuracy and macro-F1, why cross-validation was necessary, why class weights are on by
default, and why the original macro-F1 ≥ 0.92 target was abandoned.

Every run logs a SHA-256 fingerprint of each split it consumed
(`data.train.sha256` and friends), so two runs that disagree on metrics can be checked
for whether they even saw the same data. All runs below share fingerprint
`006759bf5ccc43dc…`.

## Baseline: TF-IDF + LogisticRegression

Held out on the test split, 5-fold stratified cross-validation on train:

| Metric | Cross-validated | Held-out test |
|---|---|---|
| macro-F1 | 0.7138 ± 0.0043 | **0.7149** |
| accuracy | 0.8766 ± 0.0036 | **0.8629** |
| macro-precision | 0.7014 ± 0.0040 | 0.7047 |
| macro-recall | 0.7412 ± 0.0058 | 0.7358 |

**The 15-point gap between accuracy and macro-F1 is the `neutral` class.** Accuracy is
dominated by the two large classes; macro-F1 weights `neutral` equally with them. A
model can look 86% right and still be poor at a third of the label space. This is why the
promotion gate requires both.

Cross-validation was not a formality. With `neutral` at 458 rows, a single split can
shift macro-F1 by several points on that class alone — the standard deviation above
(±0.004) is what makes any later comparison believable rather than lucky.

## Hyperparameter search

10 Optuna trials (TPE sampler, seeded), each a nested MLflow run under the parent, each
optimising **cross-validated** macro-F1 rather than a single held-out score.

| | Value |
|---|---|
| Trials | 10 |
| Objective | 3-fold CV macro-F1 |
| Range across trials | **0.5975 – 0.7079** |
| Best | trial 0, 0.7079 |
| Best params | `ngram_range=(1,2)`, `max_features=5000`, `min_df=1`, `sublinear_tf=False`, `C=0.964`, `class_weight=balanced` |

Two things worth reading off that range. An 11-point spread between the worst and best
configuration means the search was doing real work, not decorating a fixed answer. And
the best trial chose `class_weight=balanced` — the sweep independently rediscovered that
the minority class needs weighting.

Optimising cross-validated rather than single-split score costs `n_splits` times more
compute and is the only reason the ranking means anything: on a single split, the trial
that drew a favourable fold wins.

## Fairness mitigation

Three variants, measured with the identity-pair probe. Full method and per-dimension
breakdown in [`FAIRNESS.md`](FAIRNESS.md).

| Variant | Max identity gap | macro-F1 | accuracy | Gate (≤ 0.10) |
|---|---|---|---|---|
| none | 0.1417 | 0.7143 | 0.8632 | fail |
| counterfactual augmentation | 0.1080 | 0.7113 | 0.8632 | fail |
| **identity blinding** | **0.0000** | **0.7149** | 0.8629 | **pass** |

The interesting result is the middle row. Counterfactual augmentation cut the *gender*
gap by 79% (0.1417 → 0.0297) and still failed the gate, because *region* then became the
binding constraint. The reason is measurable: region terms such as
`thầy người hà nội` appear **0 times** in the training corpus, while `thầy` and `cô`
appear in 23.9% and 11.1% of rows. Augmentation can only teach the model about words it
has seen.

Blinding drives the gap to exactly zero by construction — the two halves of an identity
pair become byte-identical before vectorising — and cost nothing: macro-F1 moved
**+0.0006**. That the mitigation was free is itself the finding, and it is a measurement
rather than an assumption: it means those identity terms were carrying almost no label
signal.

## A methodology error worth recording

The first fairness measurement reported a max gap of **0.63**, which would have been
alarming. It was wrong.

The probe used capitalised honorifics — `Thầy`, `Cô` — and UIT-VSFC is entirely
lowercase: capitalised `Thầy` appears in **0.00%** of training rows. The probe was
measuring the model's behaviour on out-of-distribution casing, not bias. Lowercasing the
probe terms to match the corpus produced the 0.1417 figure above.

Recorded because the failure mode generalises: a fairness probe is an instrument, and an
instrument that has not been checked against the data distribution can manufacture a
result. It is also why `tests/responsible/` asserts that the probe detects a *known*
injected bias and reports zero for a known-fair scorer.

## What the tests locked in

Findings become regression tests, otherwise they are anecdotes:

- **Train/serve skew.** A model trained with blinding lost the flag when reloaded from
  joblib and measured a 0.08 gap instead of 0.00 — caught by the promotion gate, not by
  review. The flag now travels inside the artifact, and
  `test_blinding_survives_a_save_load_round_trip` fails if it stops doing so.
- **Held-out floor.** `test_held_out_quality_meets_the_promotion_floor` asserts
  macro-F1 ≥ 0.70 and accuracy ≥ 0.85 on the real test split, so a regression is a red
  build.
- **Calibration.** Confidence must sit in `[1/3, 1)` — never below chance for three
  classes, and never exactly 1.0, because a perfectly certain model is a miscalibrated
  one.

## Latency

| Model | p95, CPU | Budget | Status |
|---|---|---|---|
| TF-IDF baseline | **0.42 ms** | 200 ms | 476× headroom |
| XLM-RoBERTa | not measured | 200 ms | **open risk** |

The baseline's headroom has a side effect worth stating: `HighLatencyP95` cannot be made
to fire by load-testing it, because 0.42 ms will not reach 500 ms at any concurrency this
stack can generate. That alert is verified by inspection only.

## Still open: baseline vs transformer

**This is the missing half of this document, and W2-12 stays `BLOCKED` because of it.**

`xlm-roberta-base` trains and satisfies the serving contract — the `slow` test suite
loads a real checkpoint and checks it — but no fine-tuned model has been produced, so
there is nothing to compare. Fine-tuning needs a GPU; the pipeline is ready for Kaggle
(see [`KAGGLE_GUIDE.md`](KAGGLE_GUIDE.md)).

What has to be true before the comparison is worth writing:

1. **Beat the baseline on macro-F1**, target ≥ 0.80 against the baseline's 0.7149. A
   transformer that ties TF-IDF is not worth its cost.
2. **Meet the latency budget on CPU**, p95 < 200 ms. `xlm-roberta-base` is roughly four
   times DistilBERT's size and the original 200 ms target was set for DistilBERT. This is
   the largest open technical risk in the project.
3. **Pass the fairness gate**, max identity gap ≤ 0.10, re-measured on the transformer.
   Blinding is a preprocessing step and applies equally, but the number has to be
   measured rather than inherited.

If (1) holds and (2) does not, the honest outcome is to keep the baseline in `Production`
and report the transformer as an experiment that did not earn deployment. That is a
legitimate result, and the registry makes it a one-line decision rather than a rewrite.

## See also

- [`FAIRNESS.md`](FAIRNESS.md) — the probe, both mitigations, per-dimension results
- [`PROBLEM.md`](PROBLEM.md) — the success metrics these runs are measured against
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — why the pipeline is shaped this way
- [`KAGGLE_GUIDE.md`](KAGGLE_GUIDE.md) — running the transformer fine-tune on a free GPU
