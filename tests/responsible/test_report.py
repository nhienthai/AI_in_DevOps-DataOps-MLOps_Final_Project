"""The before/after report is the graded artifact, so its arithmetic is tested."""

import json
from pathlib import Path

from sentiment.responsible.fairness import FairnessResult
from sentiment.responsible.report import Measurement, build_markdown, write_report


def result(max_delta: float, mean_delta: float = 0.01) -> FairnessResult:
    return FairnessResult(
        n_sentences=216,
        n_pairs=60,
        max_delta=max_delta,
        mean_delta=mean_delta,
        max_delta_by_dimension={"gender": max_delta, "region": max_delta / 2},
        group_mean_scores={"male": 0.5, "female": 0.5 - max_delta},
        worst_pairs=[
            {
                "template": "{person} dạy rất nhiệt tình .",
                "dimension": "gender",
                "group_a": "male",
                "group_b": "female",
                "score_a": 0.6,
                "score_b": 0.6 - max_delta,
                "delta": max_delta,
            }
        ],
    )


BASELINE = Measurement("baseline", result(0.14), macro_f1=0.71, accuracy=0.86)


def test_gate_column_reflects_the_threshold() -> None:
    passing = Measurement("blinded", result(0.0), macro_f1=0.71, accuracy=0.86)
    markdown = build_markdown(BASELINE, [passing], threshold=0.10)

    assert "| baseline | 0.1400" in markdown
    assert "**fail**" in markdown
    assert "**pass**" in markdown


def test_free_mitigation_is_described_as_costless() -> None:
    better = Measurement("blinded", result(0.0), macro_f1=0.715, accuracy=0.861)
    markdown = build_markdown(BASELINE, [better], threshold=0.10)
    assert "no accuracy to pay" in markdown


def test_costly_mitigation_states_the_trade() -> None:
    costly = Measurement("blinded", result(0.02), macro_f1=0.66, accuracy=0.80)
    markdown = build_markdown(BASELINE, [costly], threshold=0.10)
    assert "at a cost of" in markdown


def test_failed_mitigation_is_still_reported() -> None:
    worse = Measurement("attempt", result(0.20), macro_f1=0.70, accuracy=0.85)
    markdown = build_markdown(BASELINE, [worse], threshold=0.10)
    assert "No strategy reduced" in markdown
    assert "negative result" in markdown


def test_best_variant_drives_the_narrative() -> None:
    """With several attempts the summary must describe the best, not the last."""
    worse = Measurement("attempt-1", result(0.20), macro_f1=0.70, accuracy=0.85)
    better = Measurement("attempt-2", result(0.0), macro_f1=0.715, accuracy=0.861)
    markdown = build_markdown(BASELINE, [worse, better], threshold=0.10)
    assert "0.1400 to 0.0000" in markdown


def test_every_variant_appears_in_the_dimension_table() -> None:
    variants = [
        Measurement("counterfactual", result(0.10), macro_f1=0.71, accuracy=0.86),
        Measurement("blinding", result(0.0), macro_f1=0.71, accuracy=0.86),
    ]
    markdown = build_markdown(BASELINE, variants, threshold=0.10)
    for label in ("baseline", "counterfactual", "blinding"):
        assert markdown.count(f"| {label} |") >= 2


def test_notes_and_limitations_are_present() -> None:
    markdown = build_markdown(BASELINE, [BASELINE], threshold=0.10, notes={"MLflow runs": "abc123"})
    assert "MLflow runs" in markdown
    assert "## Limitations" in markdown


def test_write_report_emits_markdown_and_json(tmp_path: Path) -> None:
    variant = Measurement("blinded", result(0.0), macro_f1=0.715, accuracy=0.861)
    paths = write_report(BASELINE, [variant], threshold=0.10, output_dir=tmp_path)

    assert [path.name for path in paths] == ["FAIRNESS.md", "fairness.json"]
    payload = json.loads((tmp_path / "fairness.json").read_text(encoding="utf-8"))
    assert payload["threshold"] == 0.10
    assert payload["baseline"]["max_delta"] == 0.14
    assert payload["variants"][0]["macro_f1"] == 0.715
