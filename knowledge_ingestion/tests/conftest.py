"""Shared pytest fixtures/markers for the test suite.

Most tests here are fast, hermetic unit tests built with dependency
injection (mocked boto3 clients, `docling-core`-built documents, fake
embedding/vector-store collaborators) — no network access or real ML model
required. A small number of `@pytest.mark.integration` tests exercise real
collaborators (a small real sentence-transformers model, qdrant-client's
in-memory mode) for genuine end-to-end confidence; skip them with
`pytest -m "not integration"` when offline or iterating quickly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: exercises a real (small) model and/or a real in-memory Qdrant instance"
    )


@pytest.fixture
def fixed_uuid() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def utcnow() -> datetime:
    return datetime.now(timezone.utc)
