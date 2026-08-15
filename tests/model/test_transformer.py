"""Thread safety and input formatting for the served transformer.

The concurrency test reproduces the failure that took the service down under
parallel load: Hugging Face's tokenizers mutate their own truncation and padding
configuration on every call, so two threads tokenizing at once make the Rust side
raise ``RuntimeError: Already borrowed``. The fake tokenizer below fails exactly
the same way, which keeps the regression test off the real 500 MB weights.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sentiment.models.text_format import InputFormat
from sentiment.models.transformer import TransformerPredictor


class _Encoding(dict):
    """Mapping that mimics ``BatchEncoding``: unpackable and movable to a device."""

    def to(self, _device: Any) -> "_Encoding":
        return self


class _BorrowCheckedTokenizer:
    """Stands in for a fast tokenizer that cannot be shared across threads."""

    def __init__(self) -> None:
        self.in_use = False
        self.concurrent_entries = 0
        self.seen: list[str] = []
        self._guard = threading.Lock()

    def __call__(self, texts: Any, **kwargs: Any) -> Any:
        with self._guard:
            if self.in_use:
                self.concurrent_entries += 1
                raise RuntimeError("Already borrowed")
            self.in_use = True
        try:
            time.sleep(0.01)  # widen the window a real tokenizer would race in
            self.seen.extend(texts)
            if kwargs.get("return_length"):
                return {"length": [len(t) for t in texts]}
            return _Encoding(input_ids=torch.ones((len(texts), 4), dtype=torch.long))
        finally:
            with self._guard:
                self.in_use = False


@pytest.fixture
def predictor(monkeypatch: pytest.MonkeyPatch) -> TransformerPredictor:
    """Build a predictor without touching the real weights."""
    monkeypatch.setattr(TransformerPredictor, "_load_model", lambda self: None)
    instance = TransformerPredictor(model_name_or_path="unused", model_version="test")
    instance.device = torch.device("cpu")
    instance.tokenizer = _BorrowCheckedTokenizer()
    instance.model = lambda **kwargs: SimpleNamespace(
        logits=torch.tensor([[0.1, 0.2, 0.7]] * len(kwargs["input_ids"]))
    )
    return instance


def test_predict_is_safe_under_concurrency(predictor: TransformerPredictor) -> None:
    """Two threads predicting at once must not make the tokenizer raise."""
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    def run() -> None:
        barrier.wait()
        try:
            predictor.predict(["giáo viên dạy hay", "môn học quá chán"])
        except Exception as exc:  # noqa: BLE001 - the assertion is about any failure
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent predict raised: {errors}"
    assert predictor.tokenizer.concurrent_entries == 0  # type: ignore[union-attr]


def test_predict_returns_one_result_per_text(predictor: TransformerPredictor) -> None:
    results = predictor.predict(["a", "b", "c"])
    assert len(results) == 3
    assert all(r.label == "positive" for r in results)


def test_empty_input_short_circuits(predictor: TransformerPredictor) -> None:
    assert predictor.predict([]) == []


def test_input_format_reaches_the_tokenizer(predictor: TransformerPredictor) -> None:
    """The model must receive the shape it was trained on, not the raw request."""
    predictor.input_format = InputFormat(
        clean_dataset_artifacts=True, template="Chủ đề: others | {text}"
    )
    predictor.predict(["lúc 9doubledot30"])
    assert predictor.tokenizer.seen[0] == "Chủ đề: others | lúc 9:30"  # type: ignore[union-attr]


def test_identity_format_leaves_text_untouched(predictor: TransformerPredictor) -> None:
    predictor.predict(["giáo viên dạy hay"])
    assert predictor.tokenizer.seen[0] == "giáo viên dạy hay"  # type: ignore[union-attr]
