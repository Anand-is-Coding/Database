"""Custom exceptions for the ingestion orchestrator.

Deliberately small: per-document failures are caught and recorded in
`PipelineState` rather than raised (see `IngestionPipeline.run_document`),
so these are reserved for orchestrator-level problems that aren't tied to
any single document.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all orchestrator-level failures."""


class CheckpointError(PipelineError):
    """The checkpoint state file could not be loaded or saved."""
