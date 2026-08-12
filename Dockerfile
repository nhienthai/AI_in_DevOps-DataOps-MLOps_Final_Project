# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS builder

ARG BUILD_REVISION=unknown

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-serving.txt pyproject.toml ./
RUN pip install --upgrade pip==24.3.1 \
    && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --requirement requirements-serving.txt

COPY src ./src
RUN pip install --no-deps .

FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS runtime

ARG BUILD_REVISION=unknown

LABEL org.opencontainers.image.title="sentiment-service" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.source="https://github.com/nhienthai/AI_in_DevOps-DataOps-MLOps_Final_Project"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_BUILD_REVISION="${BUILD_REVISION}"

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=2)"]

CMD ["uvicorn", "sentiment.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
