"""Custom exceptions for the PDF parsing stage.

Kept local to `parser/` so callers only need to handle a small, well-defined
set of errors instead of Docling/PDF-backend internals leaking through.
"""

from __future__ import annotations


class ParserError(Exception):
    """Base exception for all PDF parsing failures."""


class CorruptedPDFError(ParserError):
    """The PDF file is malformed or unreadable by the parsing backend."""


class PasswordProtectedPDFError(ParserError):
    """The PDF is encrypted and cannot be opened without a password."""


class UnsupportedDocumentError(ParserError):
    """The file is not a PDF, or uses a feature the parsing backend cannot handle."""


class EmptyDocumentError(ParserError):
    """The PDF contains no pages or no extractable content."""


class DocumentValidationError(ParserError):
    """A parsed document failed post-parse structural validation."""
