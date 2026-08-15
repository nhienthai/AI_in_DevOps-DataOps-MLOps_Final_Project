"""Input shaping that must match between training and serving.

A model only sees what its tokenizer was fed. The PhoBERT-v2 run trains on
``"Chủ đề: {topic} | {cleaned sentence}"`` rather than the bare sentence, so
serving a raw sentence to it costs real accuracy — measured on the 3166-example
UIT-VSFC test set, macro-F1 falls from 0.8457 to 0.7956. The transformation
therefore travels with the model instead of living in a training notebook.

Kept dependency-free (``re`` only) so the serving image, which has no
``datasets`` or ``pandas``, can import it.
"""

import re
from dataclasses import dataclass

# UIT-VSFC ships these placeholders in place of punctuation and anonymised ids.
_ARTIFACT_SUBSTITUTIONS = (
    (re.compile(r"doubledot", re.IGNORECASE), ":"),
    (re.compile(r"\bfraction\b", re.IGNORECASE), "/"),
    (re.compile(r"wzjwz\d+", re.IGNORECASE), "[ANON]"),
)

TEXT_PLACEHOLDER = "{text}"


def clean_text_vietnamese(text: str) -> str:
    """Restore the punctuation UIT-VSFC encoded as words."""
    if not isinstance(text, str):
        return ""
    for pattern, replacement in _ARTIFACT_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


@dataclass(frozen=True)
class InputFormat:
    """How raw request text becomes model input.

    Attributes:
        clean_dataset_artifacts: Apply :func:`clean_text_vietnamese` first.
        template: A string containing ``{text}``, or ``None`` to pass the text
            through unchanged. Topic-conditioned models use a constant topic here
            because the HTTP contract carries no topic field.
    """

    clean_dataset_artifacts: bool = False
    template: str | None = None

    def __post_init__(self) -> None:
        if self.template is not None and TEXT_PLACEHOLDER not in self.template:
            raise ValueError(f"input template must contain {TEXT_PLACEHOLDER}: {self.template!r}")

    @property
    def is_identity(self) -> bool:
        """True when the format leaves request text untouched."""
        return not self.clean_dataset_artifacts and self.template is None

    def apply(self, text: str) -> str:
        """Shape one request string into what the model was trained on."""
        if self.clean_dataset_artifacts:
            text = clean_text_vietnamese(text)
        if self.template is not None:
            text = self.template.replace(TEXT_PLACEHOLDER, text)
        return text

    @classmethod
    def from_metadata(cls, payload: object) -> "InputFormat":
        """Build from the ``preprocessing`` block of ``serving_metadata.json``.

        Anything unrecognised yields the identity format: a model whose metadata
        is missing or malformed still serves, just without the extra shaping.
        """
        if not isinstance(payload, dict):
            return cls()
        template = payload.get("template")
        if template is not None and (
            not isinstance(template, str) or TEXT_PLACEHOLDER not in template
        ):
            return cls()
        return cls(
            clean_dataset_artifacts=bool(payload.get("clean_dataset_artifacts", False)),
            template=template,
        )
