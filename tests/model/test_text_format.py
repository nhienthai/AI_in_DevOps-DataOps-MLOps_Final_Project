"""Input shaping shared by training and serving."""

import pytest

from sentiment.models.text_format import InputFormat, clean_text_vietnamese


def test_clean_restores_encoded_punctuation() -> None:
    assert clean_text_vietnamese("lớp bắt đầu 11doubledot55") == "lớp bắt đầu 11:55"
    assert clean_text_vietnamese("điểm 8 fraction 10") == "điểm 8 / 10"
    assert clean_text_vietnamese("thầy wzjwz123 dạy hay") == "thầy [ANON] dạy hay"


def test_clean_tolerates_non_strings() -> None:
    assert clean_text_vietnamese(None) == ""  # type: ignore[arg-type]


def test_identity_format_passes_text_through() -> None:
    fmt = InputFormat()
    assert fmt.is_identity
    assert fmt.apply("giáo viên dạy hay") == "giáo viên dạy hay"


def test_template_wraps_the_text() -> None:
    fmt = InputFormat(template="Chủ đề: others | {text}")
    assert not fmt.is_identity
    assert fmt.apply("dạy hay") == "Chủ đề: others | dạy hay"


def test_template_and_cleaning_compose_in_order() -> None:
    fmt = InputFormat(clean_dataset_artifacts=True, template="Chủ đề: others | {text}")
    assert fmt.apply("  lúc 9doubledot30  ") == "Chủ đề: others | lúc 9:30"


def test_template_without_placeholder_is_rejected() -> None:
    """A template that drops the text would silently classify a constant."""
    with pytest.raises(ValueError, match=r"\{text\}"):
        InputFormat(template="Chủ đề: others")


@pytest.mark.parametrize(
    "payload",
    [None, "nonsense", {}, {"template": "no placeholder"}, {"template": 42}],
)
def test_unusable_metadata_falls_back_to_identity(payload: object) -> None:
    assert InputFormat.from_metadata(payload).is_identity


def test_metadata_round_trip() -> None:
    fmt = InputFormat.from_metadata(
        {"clean_dataset_artifacts": True, "template": "Chủ đề: others | {text}"}
    )
    assert fmt.clean_dataset_artifacts is True
    assert fmt.apply("dạy hay") == "Chủ đề: others | dạy hay"
