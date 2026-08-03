"""Custom exceptions for the chunking stage.

Kept local to `chunking/` so callers only need to handle a small,
well-defined set of errors instead of validation internals leaking through.
"""

from __future__ import annotations


class ChunkingError(Exception):
    """Base exception for all chunking failures."""


class ChunkValidationError(ChunkingError):
    """Generated chunks failed post-chunking structural validation."""
