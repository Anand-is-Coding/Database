"""Custom exceptions for the retrieval stage.

Kept local to `retrieval/` so callers only need to handle a small,
well-defined set of errors instead of qdrant-client/sentence-transformers
internals leaking through.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base exception for all retrieval failures."""


class CollectionNotFoundError(RetrievalError):
    """No Qdrant collection exists for the requested subject."""


class QueryEmbeddingError(RetrievalError):
    """The student's question could not be embedded."""


class SearchError(RetrievalError):
    """A Qdrant search failed after all retries were exhausted."""


class SearchTimeoutError(RetrievalError):
    """A search did not complete within the configured timeout."""


class RetrievalValidationError(RetrievalError):
    """Retrieved results failed post-search structural validation."""
