"""Identity-pair fairness probing, run over HTTP against the deployed service.

The method follows the Equity Evaluation Corpus (Kiritchenko & Mohammad, 2018).
UIT-VSFC carries no demographic columns, so group-wise accuracy is impossible to
compute — there is nobody to group by. Instead the probe interrogates the model
directly: build pairs of sentences that are identical except for one identity
term, and measure whether the sentiment the system reports changes. The question
is falsifiable, which is what makes the resulting number defensible.

The corpus is adapted to Vietnamese student feedback rather than translated. The
English EEC varies given names by race and gender; the equivalent axes in this
domain are the honorifics Vietnamese students use for teachers, given names, and
regional terms.

Two rules the implementation follows deliberately:

1. **The probe never imports the model.** It reaches the running service over
   HTTP, so the measurement covers the deployed preprocessing, truncation and
   checkpoint. A number produced by importing the model would describe a
   different system from the one users talk to.
2. **The score compared is the model's continuous output**, not the argmax label.
   Two sentences can share a label while the model is visibly less positive
   about one of them, and that gap is the bias worth reporting.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence

ScoreFn = Callable[[Sequence[str]], list[float]]

TEMPLATES: tuple[str, ...] = (
    "{person} dạy rất nhiệt tình .",
    "{person} giảng bài rất dễ hiểu .",
    "{person} chuẩn bị bài giảng rất công phu .",
    "em học được nhiều điều từ {person} .",
    "{person} luôn giải đáp thắc mắc của sinh viên .",
    "{person} giảng bài khó hiểu .",
    "{person} chấm điểm không công bằng .",
    "em không theo được nhịp độ giảng của {person} .",
    "{person} thường đến lớp muộn .",
    "{person} dạy đúng theo giáo trình .",
    "{person} phụ trách môn học này .",
    "lớp của {person} có bốn mươi sinh viên .",
)


@dataclass(frozen=True)
class IdentityGroup:
    """One group of interchangeable identity terms along a single axis."""

    name: str
    dimension: str
    terms: tuple[str, ...]


GROUPS: tuple[IdentityGroup, ...] = (
    IdentityGroup("male", "gender", ("thầy", "thầy nam", "thầy tuấn", "anh hùng")),
    IdentityGroup("female", "gender", ("cô", "cô lan", "cô hương", "chị mai")),
    IdentityGroup("northern", "region", ("thầy người hà nội", "cô người miền bắc")),
    IdentityGroup("central", "region", ("thầy người huế", "cô người miền trung")),
    IdentityGroup("southern", "region", ("thầy người sài gòn", "cô người miền nam")),
    IdentityGroup("senior", "seniority", ("giáo sư", "giảng viên chính")),
    IdentityGroup("junior", "seniority", ("trợ giảng", "giảng viên tập sự")),
)


@dataclass(frozen=True)
class PairDelta:
    """Score gap between two groups on one template."""

    template: str
    dimension: str
    group_a: str
    group_b: str
    score_a: float
    score_b: float

    @property
    def delta(self) -> float:
        """Absolute score gap."""
        return abs(self.score_a - self.score_b)


@dataclass
class FairnessResult:
    """Outcome of one probe run."""

    n_sentences: int
    n_pairs: int
    max_delta: float
    mean_delta: float
    max_delta_by_dimension: dict[str, float]
    group_mean_scores: dict[str, float]
    worst_pairs: list[dict[str, Any]] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float]:
        """Return a flat mapping suitable for ``mlflow.log_metrics``."""
        metrics = {
            "fairness_max_delta": self.max_delta,
            "fairness_mean_delta": self.mean_delta,
        }
        for dimension, value in self.max_delta_by_dimension.items():
            metrics[f"fairness_max_delta_{dimension}"] = value
        for group, value in self.group_mean_scores.items():
            metrics[f"fairness_group_score_{group}"] = value
        return metrics

    def passes(self, threshold: float) -> bool:
        """Whether the worst identity-pair gap is within ``threshold``."""
        return self.max_delta <= threshold


def build_sentences(
    templates: Iterable[str] = TEMPLATES, groups: Iterable[IdentityGroup] = GROUPS
) -> list[tuple[str, str, str, str]]:
    """Return ``(template, group, term, sentence)`` for the whole probe set."""
    rows: list[tuple[str, str, str, str]] = []
    for template in templates:
        for group in groups:
            for term in group.terms:
                rows.append((template, group.name, term, template.format(person=term)))
    return rows


def probe(
    score_fn: ScoreFn,
    templates: Iterable[str] = TEMPLATES,
    groups: Iterable[IdentityGroup] = GROUPS,
) -> FairnessResult:
    """Measure identity-pair score gaps using ``score_fn`` to score sentences.

    ``score_fn`` receives a batch of sentences and returns one score per
    sentence. Passing a function rather than a URL keeps this testable in-process
    while the production path stays HTTP-only.

    Args:
        score_fn: Batch scoring function.
        templates: Sentence templates carrying a ``{person}`` slot.
        groups: Identity groups to compare.

    Returns:
        A :class:`FairnessResult` summarising the run.

    Raises:
        ValueError: If ``score_fn`` returns the wrong number of scores.
    """
    group_list = list(groups)
    template_list = list(templates)
    rows = build_sentences(template_list, group_list)
    scores = score_fn([sentence for _, _, _, sentence in rows])
    if len(scores) != len(rows):
        raise ValueError(f"score_fn returned {len(scores)} scores for {len(rows)} sentences")

    by_template_group: dict[tuple[str, str], list[float]] = {}
    by_group: dict[str, list[float]] = {}
    for (template, group, _, _), score in zip(rows, scores):
        by_template_group.setdefault((template, group), []).append(score)
        by_group.setdefault(group, []).append(score)

    deltas: list[PairDelta] = []
    for template in template_list:
        for group_a, group_b in itertools.combinations(group_list, 2):
            if group_a.dimension != group_b.dimension:
                continue
            scores_a = by_template_group.get((template, group_a.name))
            scores_b = by_template_group.get((template, group_b.name))
            if not scores_a or not scores_b:
                continue
            deltas.append(
                PairDelta(
                    template=template,
                    dimension=group_a.dimension,
                    group_a=group_a.name,
                    group_b=group_b.name,
                    score_a=fmean(scores_a),
                    score_b=fmean(scores_b),
                )
            )

    if not deltas:
        raise ValueError("no comparable identity pairs were produced")

    by_dimension: dict[str, float] = {}
    for delta in deltas:
        current = by_dimension.get(delta.dimension, 0.0)
        by_dimension[delta.dimension] = max(current, delta.delta)

    ranked = sorted(deltas, key=lambda item: item.delta, reverse=True)
    return FairnessResult(
        n_sentences=len(rows),
        n_pairs=len(deltas),
        max_delta=ranked[0].delta,
        mean_delta=fmean(delta.delta for delta in deltas),
        max_delta_by_dimension=by_dimension,
        group_mean_scores={group: fmean(values) for group, values in sorted(by_group.items())},
        worst_pairs=[
            {
                "template": delta.template,
                "dimension": delta.dimension,
                "group_a": delta.group_a,
                "group_b": delta.group_b,
                "score_a": round(delta.score_a, 6),
                "score_b": round(delta.score_b, 6),
                "delta": round(delta.delta, 6),
            }
            for delta in ranked[:10]
        ],
    )


def http_score_fn(base_url: str, batch_size: int = 32, timeout: float = 30.0) -> ScoreFn:
    """Return a scoring function that calls ``POST /api/v1/predict/batch``.

    Args:
        base_url: Base URL of the running service.
        batch_size: Sentences per request, bounded by the server's own limit.
        timeout: Per-request timeout in seconds.

    Returns:
        A callable suitable for :func:`probe`.
    """
    import httpx

    def score(texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                response = client.post("/api/v1/predict/batch", json={"texts": chunk})
                response.raise_for_status()
                for item in response.json()["results"]:
                    prediction = item.get("prediction")
                    if prediction is None:
                        raise RuntimeError(
                            f"service rejected a probe sentence: {item.get('error')}"
                        )
                    scores.append(float(prediction["score"]))
        return scores

    return score


def probe_over_http(base_url: str, batch_size: int = 32) -> FairnessResult:
    """Run the probe against a live service."""
    return probe(http_score_fn(base_url, batch_size=batch_size))


def counterfactual_pairs(
    groups: Iterable[IdentityGroup] = GROUPS,
) -> list[tuple[str, str]]:
    """Return term substitutions that swap identity within one dimension.

    Used by the mitigation in :func:`counterfactual_augment`. Substituting only
    within a dimension keeps the sentence coherent: swapping a gender honorific
    for another gender honorific is meaningful, swapping it for a seniority term
    is not.
    """
    substitutions: list[tuple[str, str]] = []
    by_dimension: dict[str, list[IdentityGroup]] = {}
    for group in groups:
        by_dimension.setdefault(group.dimension, []).append(group)
    for members in by_dimension.values():
        for group_a, group_b in itertools.permutations(members, 2):
            for term_a, term_b in zip(group_a.terms, group_b.terms):
                substitutions.append((term_a, term_b))
    return substitutions


def counterfactual_augment(
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Iterable[IdentityGroup] = GROUPS,
    max_per_row: int = 1,
) -> tuple[list[str], list[int]]:
    """Augment training data with identity-swapped copies carrying the same label.

    This is the mitigation for W3-07. If a sentence mentioning ``Thầy`` appears
    with a positive label, the model also sees the same sentence with ``Cô`` and
    the same label, which removes the correlation between the identity term and
    the outcome that the probe detects.

    Only rows that actually contain an identity term are augmented, so the
    training set grows by the number of affected rows rather than doubling.

    Args:
        texts: Original training texts.
        labels: Labels aligned with ``texts``.
        groups: Identity groups whose terms get swapped.
        max_per_row: Maximum augmented copies generated per original row.

    Returns:
        ``(texts, labels)`` containing the originals followed by the additions.

    Raises:
        ValueError: If ``texts`` and ``labels`` differ in length.
    """
    if len(texts) != len(labels):
        raise ValueError(f"texts and labels differ: {len(texts)} vs {len(labels)}")

    substitutions = counterfactual_pairs(groups)
    out_texts = list(texts)
    out_labels = list(labels)

    for text, label in zip(texts, labels):
        added = 0
        for source, target in substitutions:
            if added >= max_per_row:
                break
            if not source:
                continue
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            swapped, count = pattern.subn(target, text)
            if count and swapped != text:
                out_texts.append(swapped)
                out_labels.append(label)
                added += 1

    return out_texts, out_labels


IDENTITY_PLACEHOLDER = "người dạy"


def identity_terms(groups: Iterable[IdentityGroup] = GROUPS) -> list[str]:
    """Return every identity term, longest first so replacement is unambiguous.

    Ordering matters: replacing ``thầy`` before ``thầy người hà nội`` would leave
    ``người dạy người hà nội`` behind and defeat the point.
    """
    terms = [term for group in groups for term in group.terms if term]
    return sorted(set(terms), key=len, reverse=True)


def blind_identity_terms(
    text: str, groups: Iterable[IdentityGroup] = GROUPS, placeholder: str = IDENTITY_PLACEHOLDER
) -> str:
    """Replace every identity term with a neutral placeholder.

    This is the second mitigation. Counterfactual augmentation teaches the model
    that identity terms do not predict the label; blinding removes the terms
    before the model can see them at all, which makes the identity-pair delta
    zero by construction rather than by training.

    The trade-off is real and worth stating: any genuine signal those terms carry
    is destroyed along with the bias. It is only the right choice when the terms
    carry little signal, which has to be measured rather than assumed.

    Critically, this must run identically at training and at serving time. A
    model trained on blinded text and served unblinded text sees a distribution
    it was never fitted on, which is why
    :class:`~sentiment.models.baseline.BaselinePredictor` applies it inside both
    ``fit`` and ``predict`` rather than leaving it to the caller.
    """
    blinded = text
    for term in identity_terms(groups):
        blinded = re.compile(re.escape(term), re.IGNORECASE).sub(placeholder, blinded)
    return blinded


def write_report(result: FairnessResult, path: Path) -> Path:
    """Write a probe result as JSON, for logging as an MLflow artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_report(path: Path) -> FairnessResult:
    """Read a probe result written by :func:`write_report`."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return FairnessResult(
        n_sentences=int(payload["n_sentences"]),
        n_pairs=int(payload["n_pairs"]),
        max_delta=float(payload["max_delta"]),
        mean_delta=float(payload["mean_delta"]),
        max_delta_by_dimension=dict(payload["max_delta_by_dimension"]),
        group_mean_scores=dict(payload["group_mean_scores"]),
        worst_pairs=list(payload.get("worst_pairs", [])),
    )
