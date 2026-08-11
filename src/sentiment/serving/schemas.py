"""Validated API contracts and OpenAPI examples."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Strict base model shared by public request schemas."""

    model_config = ConfigDict(extra="forbid")


class PredictRequest(APIModel):
    """One review to classify."""

    text: str = Field(
        min_length=1,
        description="English-language review text.",
        examples=["Arrived quickly and works perfectly."],
    )


class PredictResponse(BaseModel):
    """Sentiment assigned to one review."""

    label: Literal["positive", "negative"]
    score: float = Field(ge=0.0, le=1.0, description="P(positive).")
    confidence: float = Field(ge=0.5, le=1.0)
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
    """One batch result containing either a prediction or an error."""

    index: int = Field(ge=0)
    prediction: PredictResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Ordered per-item batch results."""

    results: list[BatchItem]


class ModelInfo(BaseModel):
    """Provenance and evaluation metadata for the serving model."""

    model_version: str
    stage: str
    predictor_class: str
    metrics: dict[str, float] = Field(default_factory=dict)
    fairness_delta: float | None = None
    trained_at: str | None = None
    run_id: str | None = None


class ExplainRequest(APIModel):
    """One review and the requested local explanation method."""

    text: str = Field(min_length=1, examples=["The battery life is excellent."])
    method: Literal["lime"] = "lime"


class Attribution(BaseModel):
    """One token and its signed contribution."""

    token: str
    attribution: float


class ExplainResponse(BaseModel):
    """Local token-level explanation of one prediction."""

    method: Literal["lime"]
    label: Literal["positive", "negative"]
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    attributions: list[Attribution]


class ErrorResponse(BaseModel):
    """Uniform error returned by all failing endpoints."""

    error_code: str
    message: str
    request_id: str
