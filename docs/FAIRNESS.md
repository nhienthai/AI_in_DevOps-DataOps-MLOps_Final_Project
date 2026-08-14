# Fairness: before and after mitigation

Generated 2026-08-13 16:17 UTC. A model is promotable only when its worst identity-pair gap is **≤ 0.10**; `scripts/validate_model.py` enforces this, so fairness is a build failure rather than a paragraph.

## Before and after

| Model | Max delta | Mean delta | Macro-F1 | Accuracy | Gate |
|---|---|---|---|---|---|
| baseline (unmitigated) | 0.1417 | 0.0247 | 0.7143 | 0.8632 | **fail** |
| + counterfactual augmentation | 0.1080 | 0.0215 | 0.7113 | 0.8632 | **fail** |
| + identity blinding | 0.0000 | 0.0000 | 0.7149 | 0.8629 | **pass** |

### What it cost

The worst identity gap fell from 0.1417 to 0.0000 — a 100% reduction — while macro-F1 moved by +0.0006 and accuracy by -0.0003. There was no accuracy to pay, which is itself the finding: the identity terms were carrying almost no label signal, so removing their influence cost nothing. That is a measurement, not an assumption — a mitigation on features the model genuinely relied on would show up here as a loss.

## Worst gap by dimension

This is where the interesting result lives. A single headline number hides which axis is actually binding.

| Model | gender | region | seniority |
|---|---|---|---|
| baseline (unmitigated) | 0.1417 | 0.0868 | 0.0340 |
| + counterfactual augmentation | 0.0297 | 0.1080 | 0.0355 |
| + identity blinding | 0.0000 | 0.0000 | 0.0000 |

## Mean score by identity group

A sanity check on the deltas above: if every group moved the same way by the same amount, the model simply became more or less positive overall rather than fairer.

| Model | central | female | junior | male | northern | senior | southern |
|---|---|---|---|---|---|---|---|
| baseline (unmitigated) | 0.4517 | 0.4658 | 0.4110 | 0.4279 | 0.4513 | 0.4105 | 0.4181 |
| + counterfactual augmentation | 0.4453 | 0.4507 | 0.4133 | 0.4390 | 0.4393 | 0.4144 | 0.4038 |
| + identity blinding | 0.4731 | 0.4731 | 0.4731 | 0.4731 | 0.4731 | 0.4731 | 0.4731 |

## Method

- **Probe.** 216 sentences built from templated Vietnamese student feedback, giving 60 comparable identity pairs across gender, region and seniority. Each pair differs by exactly one identity term, so any score difference is attributable to that term.
- **Score compared** is the model's continuous positive-class probability, not the argmax label. Two sentences can share a label while the model is visibly less positive about one of them, and that gap is the bias worth reporting.
- **Two measurement points.** At training time the probe runs in-process against the candidate, because that is what gates promotion and there is no service to query yet. In production `scripts/run_fairness_probe.py` runs the same probe over HTTP against the deployed model, so the dashboard number describes what users actually talk to. Same probe set, same threshold, so the gate and the alert cannot disagree.

### Mitigations tried

1. **Counterfactual augmentation.** Every training row containing an identity term is duplicated with that term swapped within its dimension, carrying the original label. The model then sees the same sentence with `thầy` and with `cô` and the same outcome, which breaks the correlation between the term and the label.
2. **Identity blinding.** Identity terms are replaced with a neutral placeholder before vectorising, applied inside both `fit` and `predict` so a blinded model can never be served unblinded text. This drives the gap to exactly zero by construction rather than by training.

## Worst remaining pairs

| Template | Dimension | Groups | Delta |
|---|---|---|---|
| `{person} dạy rất nhiệt tình .` | gender | male vs female | 0.0000 |
| `{person} dạy rất nhiệt tình .` | region | northern vs central | 0.0000 |
| `{person} dạy rất nhiệt tình .` | region | northern vs southern | 0.0000 |
| `{person} dạy rất nhiệt tình .` | region | central vs southern | 0.0000 |
| `{person} dạy rất nhiệt tình .` | seniority | senior vs junior | 0.0000 |

## Notes

- **MLflow runs.** none=546eb453, counterfactual=f5223ae2, blinding=817cd990

## Limitations

- The probe covers the identity axes chosen here. A gap it does not measure is not evidence that no gap exists.
- Templates are synthetic. Isolating the identity term is what makes the comparison clean, but these are not sentences a real student wrote.
- Blinding removes the symptom at the input. It guarantees parity on the terms it knows about and does nothing for a term absent from its list, or for bias expressed through correlated wording rather than an explicit identity term.
- Parity across identity terms says nothing about whether the model is accurate. Read this beside macro-F1, never instead of it.
