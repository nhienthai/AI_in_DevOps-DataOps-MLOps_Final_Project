"""Validated API contracts with complete OpenAPI examples."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentiment.serving.predictor import SentimentLabel


class APIModel(BaseModel):
    """Strict base model shared by public request schemas."""

    model_config = ConfigDict(extra="forbid")


class PredictRequest(APIModel):
    """One review to classify."""

    text: str = Field(
        min_length=1,
        description="Review text to classify.",
        examples=["Giáo viên giảng bài rất dễ hiểu."],
    )


class PredictResponse(BaseModel):
    """Sentiment assigned to one review."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "positive",
                "score": 0.91,
                "confidence": 0.91,
                "model_version": "12",
                "truncated": False,
                "latency_ms": 18.42,
            }
        }
    )

    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Positive-class probability.")
    confidence: float = Field(ge=0.0, le=1.0, description="Predicted-class probability.")
    model_version: str
    truncated: bool
    latency_ms: float = Field(ge=0.0)


class BatchRequest(APIModel):
    """A non-empty collection of reviews to classify."""

    texts: list[str] = Field(
        min_length=1,
        description="One to 64 review texts; invalid items are isolated.",
        examples=[["Excellent build quality.", "Stopped working after one day."]],
    )


class BatchItem(BaseModel):
    """One batch result containing exactly one prediction or error."""

    index: int = Field(ge=0)
    prediction: PredictResponse | None = None
    error: str | None = None

    @model_validator(mode="after")
    def one_result(self) -> "BatchItem":
        if (self.prediction is None) == (self.error is None):
            raise ValueError("exactly one of prediction or error is required")
        return self


class BatchResponse(BaseModel):
    """Ordered per-item batch results."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "index": 0,
                        "prediction": {
                            "label": "positive",
                            "score": 0.91,
                            "confidence": 0.91,
                            "model_version": "12",
                            "truncated": False,
                            "latency_ms": 18.42,
                        },
                    },
                    {"index": 1, "prediction": None, "error": "Text must not be blank."},
                ]
            }
        }
    )
    results: list[BatchItem]


class ModelInfo(BaseModel):
    """Provenance and evaluation metadata for the serving model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "12",
                "stage": "Production",
                "predictor_class": "TransformerPredictor",
                "metrics": {"macro_f1": 0.89},
                "fairness_delta": 0.04,
                "trained_at": "2026-08-12T10:00:00Z",
                "run_id": "abc123",
                "build_revision": "9124bb5",
            }
        }
    )

    model_version: str
    stage: str
    predictor_class: str
    metrics: dict[str, float] = Field(default_factory=dict)
    fairness_delta: float | None = None
    trained_at: str | None = None
    run_id: str | None = None
    build_revision: str


class ExplainRequest(APIModel):
    """One review and the requested local explanation method."""

    text: str = Field(min_length=1, examples=["The battery life is excellent."])
    method: Literal["lime"] = "lime"


class Attribution(BaseModel):
    """One token and its signed contribution."""

    token: str = Field(examples=["excellent"])
    attribution: float = Field(examples=[0.42])


class ExplainResponse(BaseModel):
    """Local token-level explanation of one prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method": "lime",
                "label": "positive",
                "score": 0.91,
                "model_version": "12",
                "attributions": [{"token": "excellent", "attribution": 0.42}],
            }
        }
    )
    method: Literal["lime"]
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    attributions: list[Attribution]


class ErrorResponse(BaseModel):
    """Uniform error returned by all failing endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "text_too_long",
                "message": "Text exceeds 5000 characters.",
                "request_id": "0fdaf63f-7f50-44a9-baa2-2caab9b74001",
            }
        }
    )
    error_code: str
    message: str
    request_id: str


class ReadyResponse(BaseModel):
    """Readiness state and active model version."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ready", "model_version": "12"}}
    )
    status: Literal["ready"]
    model_version: str


class ReloadResponse(BaseModel):
    """Successful atomic model reload result."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "reloaded", "model_version": "13"}}
    )
    status: Literal["reloaded"]
    model_version: str
