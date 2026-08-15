"""Renders the fairness before/after report.

A fairness *analysis* says how biased a model is. A before/after table says what
was done about it and what it cost. The accuracy column is not decoration: a
mitigation that removes bias by making the model worse for everyone is a
different result from one that removes bias for free, and hiding the difference
would misrepresent the work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from sentiment.responsible.fairness import FairnessResult


@dataclass(frozen=True)
class Measurement:
    """One model's fairness and quality figures."""

    label: str
    fairness: FairnessResult
    macro_f1: float
    accuracy: float

    @property
    def max_delta(self) -> float:
        """Worst identity-pair gap."""
        return self.fairness.max_delta


def _format_delta(before: float, after: float, lower_is_better: bool) -> str:
    """Render a signed change with an arrow indicating whether it improved."""
    change = after - before
    if abs(change) < 5e-5:
        return "no change"
    improved = change < 0 if lower_is_better else change > 0
    arrow = "improved" if improved else "worse"
    return f"{change:+.4f} ({arrow})"


def _gate(measurement: Measurement, threshold: float) -> str:
    """Render pass/fail against the promotion threshold."""
    return "**pass**" if measurement.fairness.passes(threshold) else "**fail**"


def build_markdown(
    baseline: Measurement,
    variants: Sequence[Measurement],
    threshold: float,
    notes: Mapping[str, str] | None = None,
) -> str:
    """Render the before/after comparison as a Markdown document.

    Args:
        baseline: The unmitigated model.
        variants: One measurement per mitigation strategy tried, in the order
            they were attempted. Strategies that failed stay in the table.
        threshold: Promotion threshold applied to the max identity-pair delta.
        notes: Optional extra rows appended near the end.

    Returns:
        A complete Markdown document.
    """
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    everything = [baseline, *variants]
    dimensions = sorted(
        {key for item in everything for key in item.fairness.max_delta_by_dimension}
    )
    groups = sorted({key for item in everything for key in item.fairness.group_mean_scores})

    lines: list[str] = [
        "# Fairness: before and after mitigation",
        "",
        f"Generated {generated}. A model is promotable only when its worst identity-pair "
        f"gap is **≤ {threshold:.2f}**; `scripts/validate_model.py` enforces this, so "
        "fairness is a build failure rather than a paragraph.",
        "",
        "## Before and after",
        "",
        "| Model | Max delta | Mean delta | Macro-F1 | Accuracy | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for item in everything:
        lines.append(
            f"| {item.label} | {item.max_delta:.4f} | {item.fairness.mean_delta:.4f} | "
            f"{item.macro_f1:.4f} | {item.accuracy:.4f} | {_gate(item, threshold)} |"
        )

    best = min(variants, key=lambda item: item.max_delta) if variants else baseline
    delta_change = best.max_delta - baseline.max_delta
    f1_change = best.macro_f1 - baseline.macro_f1

    lines += [
        "",
        "### What it cost",
        "",
    ]
    if delta_change < 0 and f1_change >= 0:
        lines.append(
            f"The worst identity gap fell from {baseline.max_delta:.4f} to "
            f"{best.max_delta:.4f} — a {abs(delta_change) / max(baseline.max_delta, 1e-9):.0%} "
            f"reduction — while macro-F1 moved by {f1_change:+.4f} and accuracy by "
            f"{best.accuracy - baseline.accuracy:+.4f}. There was no accuracy to pay, which "
            "is itself the finding: the identity terms were carrying almost no label signal, "
            "so removing their influence cost nothing. That is a measurement, not an "
            "assumption — a mitigation on features the model genuinely relied on would show "
            "up here as a loss."
        )
    elif delta_change < 0:
        lines.append(
            f"The worst identity gap fell from {baseline.max_delta:.4f} to "
            f"{best.max_delta:.4f}, at a cost of {abs(f1_change):.4f} macro-F1. The trade is "
            "stated rather than hidden: the model was buying some of its accuracy from "
            "identity terms, and giving that up gives up the accuracy with it."
        )
    else:
        lines.append(
            "No strategy reduced the worst identity gap. Reported anyway — a negative result "
            "about a method is still a result, and dropping it would imply the first attempt "
            "had worked."
        )

    lines += [
        "",
        "## Worst gap by dimension",
        "",
        "This is where the interesting result lives. A single headline number hides which "
        "axis is actually binding.",
        "",
        "| Model | " + " | ".join(dimensions) + " |",
        "|---" * (len(dimensions) + 1) + "|",
    ]
    for item in everything:
        cells = " | ".join(
            f"{item.fairness.max_delta_by_dimension.get(dimension, 0.0):.4f}"
            for dimension in dimensions
        )
        lines.append(f"| {item.label} | {cells} |")

    lines += [
        "",
        "## Mean score by identity group",
        "",
        "A sanity check on the deltas above: if every group moved the same way by the same "
        "amount, the model simply became more or less positive overall rather than fairer.",
        "",
        "| Model | " + " | ".join(groups) + " |",
        "|---" * (len(groups) + 1) + "|",
    ]
    for item in everything:
        cells = " | ".join(
            f"{item.fairness.group_mean_scores.get(group, 0.0):.4f}" for group in groups
        )
        lines.append(f"| {item.label} | {cells} |")

    lines += [
        "",
        "## Method",
        "",
        f"- **Probe.** {baseline.fairness.n_sentences} sentences built from templated "
        f"Vietnamese student feedback, giving {baseline.fairness.n_pairs} comparable identity "
        "pairs across gender, region and seniority. Each pair differs by exactly one "
        "identity term, so any score difference is attributable to that term.",
        "- **Score compared** is the model's continuous positive-class probability, not the "
        "argmax label. Two sentences can share a label while the model is visibly less "
        "positive about one of them, and that gap is the bias worth reporting.",
        "- **Two measurement points.** At training time the probe runs in-process against "
        "the candidate, because that is what gates promotion and there is no service to "
        "query yet. In production `scripts/run_fairness_probe.py` runs the same probe over "
        "HTTP against the deployed model, so the dashboard number describes what users "
        "actually talk to. Same probe set, same threshold, so the gate and the alert cannot "
        "disagree.",
        "",
        "### Mitigations tried",
        "",
        "1. **Counterfactual augmentation.** Every training row containing an identity term "
        "is duplicated with that term swapped within its dimension, carrying the original "
        "label. The model then sees the same sentence with `thầy` and with `cô` and the same "
        "outcome, which breaks the correlation between the term and the label.",
        "2. **Identity blinding.** Identity terms are replaced with a neutral placeholder "
        "before vectorising, applied inside both `fit` and `predict` so a blinded model can "
        "never be served unblinded text. This drives the gap to exactly zero by "
        "construction rather than by training.",
        "",
        "## Worst remaining pairs",
        "",
        "| Template | Dimension | Groups | Delta |",
        "|---|---|---|---|",
    ]
    for pair in best.fairness.worst_pairs[:5]:
        template = str(pair["template"]).replace("|", "\\|")
        lines.append(
            f"| `{template}` | {pair['dimension']} | "
            f"{pair['group_a']} vs {pair['group_b']} | {float(pair['delta']):.4f} |"
        )

    if notes:
        lines += ["", "## Notes", ""]
        for key, value in notes.items():
            lines.append(f"- **{key}.** {value}")

    lines += [
        "",
        "## Limitations",
        "",
        "- The probe covers the identity axes chosen here. A gap it does not measure is not "
        "evidence that no gap exists.",
        "- Templates are synthetic. Isolating the identity term is what makes the comparison "
        "clean, but these are not sentences a real student wrote.",
        "- Blinding removes the symptom at the input. It guarantees parity on the terms it "
        "knows about and does nothing for a term absent from its list, or for bias expressed "
        "through correlated wording rather than an explicit identity term.",
        "- Parity across identity terms says nothing about whether the model is accurate. "
        "Read this beside macro-F1, never instead of it.",
        "",
    ]
    return "\n".join(lines)


def _payload(measurement: Measurement) -> dict[str, object]:
    """Serialise one measurement."""
    return {
        "label": measurement.label,
        "max_delta": measurement.max_delta,
        "mean_delta": measurement.fairness.mean_delta,
        "macro_f1": measurement.macro_f1,
        "accuracy": measurement.accuracy,
        "max_delta_by_dimension": measurement.fairness.max_delta_by_dimension,
        "group_mean_scores": measurement.fairness.group_mean_scores,
    }


def write_report(
    baseline: Measurement,
    variants: Sequence[Measurement],
    threshold: float,
    output_dir: Path,
    notes: Mapping[str, str] | None = None,
    stem: str = "FAIRNESS",
) -> list[Path]:
    """Write the Markdown report and its machine-readable companion.

    Returns:
        Paths written, for handing to ``mlflow.log_artifacts``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(build_markdown(baseline, variants, threshold, notes), encoding="utf-8")

    payload = {
        "threshold": threshold,
        "baseline": _payload(baseline),
        "variants": [_payload(item) for item in variants],
    }
    json_path = output_dir / f"{stem.lower()}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return [markdown_path, json_path]
