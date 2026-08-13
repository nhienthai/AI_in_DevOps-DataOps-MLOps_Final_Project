"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a client whose context runs the application lifespan."""
    from fastapi.testclient import TestClient

    from sentiment.serving.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
