"""Custom exceptions for the Qdrant vector-store stage.

Kept local to `vectorstore/` so callers only need to handle a small,
well-defined set of errors instead of qdrant-client/httpx internals
leaking through.
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """Base exception for all vector-store failures."""


class QdrantConnectionError(VectorStoreError):
    """Qdrant could not be reached (network failure, timeout)."""


class QdrantAuthenticationError(VectorStoreError):
    """Qdrant rejected the request due to a missing/invalid API key."""


class CollectionCreationError(VectorStoreError):
    """A collection could not be created (or its config could not be verified)."""


class BatchUploadError(VectorStoreError):
    """A batch of points failed to upload after all retries were exhausted."""


class InvalidPayloadError(VectorStoreError):
    """A chunk's vector or payload is malformed and cannot be stored."""


class QdrantTimeoutError(VectorStoreError):
    """A Qdrant operation did not complete within the configured timeout."""
