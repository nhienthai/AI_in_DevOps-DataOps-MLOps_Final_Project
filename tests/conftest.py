"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sentiment.serving.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a client whose context runs the application lifespan."""
    with TestClient(create_app()) as test_client:
        yield test_client
