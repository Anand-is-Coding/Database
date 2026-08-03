"""Custom exceptions for the embedding stage.

Kept local to `embedding/` so callers only need to handle a small,
well-defined set of errors instead of sentence-transformers/torch
internals leaking through.
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for all embedding failures."""


class ModelLoadError(EmbeddingError):
    """The embedding model failed to load (download failure, bad name, OOM)."""


class EmptyChunkTextError(EmbeddingError):
    """One or more chunks have empty/whitespace-only text."""


class InvalidVectorError(EmbeddingError):
    """A generated vector has the wrong shape, or contains NaN/Inf values."""


class EmbeddingGenerationError(EmbeddingError):
    """The model failed to generate embeddings for a batch."""


class EmbeddingTimeoutError(EmbeddingError):
    """A batch did not finish embedding within the configured timeout."""


class EmbeddingValidationError(EmbeddingError):
    """Generated embeddings failed post-embedding structural validation."""
