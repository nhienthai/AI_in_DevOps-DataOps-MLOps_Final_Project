# Ethics, privacy and failure modes

This service reads what students wrote about their teachers and assigns a sentiment
label. That is not a neutral act. The document states what the system does with
personal data, where it will be wrong, and which uses it should refuse — before
anyone asks.

## 1. What the data is

UIT-VSFC is 16,175 Vietnamese sentences of real student feedback about university
courses and lecturers, published for research by UIT. It is free text written by
identifiable people about identifiable people.

**It contains no demographic columns.** That absence shapes the whole fairness
approach: group-wise accuracy is impossible because there is nobody to group by, which
is why fairness is measured by probing the model with identity-swapped sentences
instead (see [`FAIRNESS.md`](FAIRNESS.md)).

## 2. Privacy

### What the service stores

Nothing. There is no database of requests. Predictions are computed and returned; the
text is not written to disk, not logged, and not retained after the response.

Three deliberate design choices enforce that:

- **Request bodies are never logged.** The access log is disabled entirely
  (`uvicorn --no-access-log`), and the structured logs carry status codes, latency and
  the model version — never the input text. A log line is a copy of the data, and a
  copy retained indefinitely is a copy that will eventually leak.
- **Metrics are aggregates only.** `ml_input_length_chars` records how long inputs are;
  `ml_prediction_confidence` records how sure the model was. Neither can reconstruct a
  sentence. The rolling drift window holds lengths and confidences, not text.
- **The container is stateless and read-only.** `read_only: true` with a `tmpfs` for
  `/tmp` means the process physically cannot persist anything across a restart.

### What is retained

The training corpus, in the artifact volume, as the dataset that produced a given model
version. This is a research dataset published for the purpose, and the fingerprint in
`data/version.py` records exactly which rows a model saw — which is a privacy property
as well as a reproducibility one: you can answer "what data is in this model" without
guessing.

### PII in free text

Students sometimes name people: *"thầy Tuấn chấm điểm không công bằng"*. That name
arrives in the request body and reaches the model. Today the service does not scrub it,
because scrubbing it would be the wrong trade for a system that does not store it: an
imperfect scrubber that silently mangles input is worse than no scrubber when nothing
is retained anyway.

**This becomes a requirement the moment anything is stored.** If a feedback endpoint or
request logging is ever added — both are listed as future work — PII scrubbing stops
being optional and has to land in the same change, not after it.

## 3. Fairness

Measured, mitigated and gated rather than asserted. The numbers, the two mitigation
strategies tried, and the accuracy cost of each are in [`FAIRNESS.md`](FAIRNESS.md).

Three things worth stating in ethical rather than numerical terms:

**The system was biased before it was fixed.** The unmitigated baseline scored the same
sentence 0.14 more positively depending on whether it said `thầy` or `cô`. That is a
model that would systematically rate feedback about female lecturers differently from
identical feedback about male ones.

**Blinding removes the symptom, not the cause.** Identity terms are stripped before the
model sees them, which guarantees parity on the terms in the list. It does nothing about
a term nobody thought of, and nothing about bias expressed through correlated wording
rather than an explicit identity word. Reporting a delta of 0.0000 without that caveat
would be misleading.

**Parity is not accuracy.** A model that assigns every input the same label has perfect
identity parity and is useless. The fairness number is only meaningful read beside
macro-F1, which is why the promotion gate checks both.

## 4. Where this model will be wrong

Stated concretely, because "the model may make mistakes" tells nobody anything.

| Failure mode | Why it happens | What it looks like |
|---|---|---|
| **Sarcasm and irony** | Sentiment models key on lexical polarity; irony inverts meaning without inverting words. *"Thầy dạy hay lắm, cả lớp trượt hết."* | Confidently positive, factually negative |
| **Negation and concession** | Bag-of-words baselines cannot represent scope. *"không hay"* shares tokens with *"hay"* | The LIME output in the README shows the model reading `nhưng` correctly — it does not always |
| **Minority class collapse** | `neutral` is 4% of training data. Macro-F1 0.71 against accuracy 0.86 is exactly this gap | Genuinely neutral feedback pushed to positive or negative |
| **Code-switching and slang** | Student writing mixes English, abbreviations and regional slang not in the training vocabulary | Low confidence, or confident nonsense |
| **Out-of-domain input** | The model knows course feedback. Given a product review or a news headline it will still answer | A fluent, confident, meaningless label |
| **Long input** | Truncated at 256 tokens; the sentiment may live in the discarded part | `truncated: true` in the response — check it |

The service exposes `confidence` and a `truncated` flag on every response, and counts
low-confidence predictions in `ml_low_confidence_total`, so these failures are
observable rather than silent. That is the mitigation: not preventing the errors, but
refusing to hide them.

## 5. Uses this system should refuse

The intended use is **aggregate, advisory triage**: summarising thousands of feedback
comments so a department can see trends and route the negative ones to a human faster.

It must not be used for:

- **Evaluating an individual lecturer.** A per-teacher sentiment score computed from a
  model with a measured identity gap, on sarcasm it cannot read, is a number that looks
  objective and is not. Employment consequences from such a score would be
  indefensible.
- **Any automated decision about a person** — promotion, pay, contract renewal,
  discipline — with no human review.
- **Identifying or acting against individual students** based on what they wrote.
  Feedback is given on an expectation of low stakes; using it to single people out
  breaks that expectation whether or not it is technically permitted.
- **Domains it was not trained on.** It will answer confidently anywhere, and be
  reliable only on Vietnamese course feedback.

## 6. Accountability

- **Every prediction is traceable.** The response carries `model_version`; the version
  resolves to an MLflow run; the run carries its params, metrics, dataset fingerprint,
  fairness measurement and SHAP importance. "Why did the system say that in March" is
  an answerable question.
- **Every promotion passes a gate.** `scripts/validate_model.py` refuses to promote on
  macro-F1, accuracy, latency **or** fairness. Bias is a build failure, not a
  discussion.
- **Every deployed model is watched.** `FairnessRegression` fires if a serving model
  exceeds the same threshold the gate applies, so a model that got past the gate does
  not stay unnoticed.
- **Explanations are available per request.** `POST /api/v1/explain` returns token
  attributions, so a disputed prediction can be examined rather than defended by
  authority.

## 7. Environmental cost

Small and worth stating rather than ignoring. The baseline trains in seconds on a CPU.
The transformer is fine-tuned on Kaggle's free tier, in the order of one GPU-hour per
run. Inference is CPU-only, single replica. The largest avoidable cost would be
repeated hyperparameter sweeps on the transformer, which is why the Optuna sweep
targets the baseline and the transformer is trained deliberately rather than swept.

## See also

- [`FAIRNESS.md`](FAIRNESS.md) — the measured bias, both mitigations, and the cost
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — where each guarantee is implemented
- [`user-guide.md`](user-guide.md) — the alerts that make these properties observable
