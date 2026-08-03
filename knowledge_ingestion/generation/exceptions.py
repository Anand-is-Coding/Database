"""Custom exceptions for the generation stage.

Kept local to `generation/` so callers only need to handle a small,
well-defined set of errors instead of the Gemini SDK's internals leaking
through.
"""

from __future__ import annotations


class GenerationError(Exception):
    """Base exception for all answer-generation failures."""


class EmptyContextError(GenerationError):
    """No chunks were retrieved to ground an answer in."""


class AnswerGenerationError(GenerationError):
    """The LLM failed to produce an answer."""
