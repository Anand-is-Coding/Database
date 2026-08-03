"""Configuration, checkpoint, and metrics models for the ingestion orchestrator.

`PipelineState` is the persisted checkpoint that makes a run resumable —
everything else here is either input configuration or output metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from pipeline.exceptions import CheckpointError


class PipelineConfig(BaseModel):
    """Tunable knobs for `IngestionPipeline`."""

    overwrite_existing: bool = False
    max_retries: int = Field(default=3, ge=1)
    batch_size: int = Field(default=100, gt=0)
    parallel_workers: int = Field(default=1, ge=1)
    dry_run: bool = False
    download_root: Path = Path("downloads")
    checkpoint_path: Path = Path("pipeline_state.json")
    show_progress: bool = True
    checkpoint_save_interval: int = Field(
        default=1,
        ge=1,
        description=(
            "Persist PipelineState to disk every N completed documents "
            "(plus always at the end of a run). The default of 1 preserves "
            "the strongest crash-safety guarantee (at most one document's "
            "progress can ever be lost). PipelineState.save() rewrites the "
            "*entire* checkpoint file each time, so on very large runs "
            "(tens/hundreds of thousands of documents) that cost grows with "
            "how many documents have completed so far - raising this value "
            "trades a bounded amount of crash-safety (re-processing, not "
            "corruption, is the worst case, since every downstream write is "
            "idempotent) for meaningfully less I/O."
        ),
    )


class PipelineStage(str, Enum):
    """Where a document currently is (or last was) in the 7-step workflow."""

    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentState(BaseModel):
    """Checkpoint record for a single S3 PDF, keyed by its S3 key.

    The S3 key (stable from the moment a PDF is discovered) is the
    checkpoint identity - `document_id` is only assigned once the document
    has been parsed at least once, so it's recorded once known rather than
    used as the primary key.
    """

    s3_key: str
    subject: str
    document_id: str | None = None
    etag: str | None = None
    stage: PipelineStage = PipelineStage.DISCOVERED
    success: bool = False
    last_error: str | None = None
    retry_count: int = 0
    first_attempted_at: datetime | None = None
    last_attempted_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_seconds: float | None = None


class PipelineState(BaseModel):
    """The full persisted checkpoint for a pipeline's run history.

    Saved to `PipelineConfig.checkpoint_path` (by default after every
    document, see `PipelineConfig.checkpoint_save_interval`), so an
    interrupted run resumes from the last completed document rather than
    starting over.
    """

    documents: dict[str, DocumentState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load a checkpoint from disk, or return a fresh empty state if none exists yet."""
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - any parse/IO failure maps to one error type
            raise CheckpointError(f"Failed to load checkpoint state from '{path}': {exc}") from exc

    def save(self, path: Path) -> None:
        """Persist the checkpoint atomically (write to a temp file, then replace).

        Written compact (no indentation) rather than pretty-printed: this
        file is machine-read only (via `load()`), and at tens/hundreds of
        thousands of documents the indentation overhead alone is a
        meaningful fraction of an otherwise-avoidable rewrite cost.
        """
        self.updated_at = datetime.now(timezone.utc)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f"{path.name}.tmp"
            tmp_path.write_text(self.model_dump_json(), encoding="utf-8")
            tmp_path.replace(path)
        except Exception as exc:  # noqa: BLE001 - any IO failure maps to one error type
            raise CheckpointError(f"Failed to save checkpoint state to '{path}': {exc}") from exc


class PipelineStatistics(BaseModel):
    """Metrics collected over one `run()`/`run_subject()` call."""

    total_pdfs: int = 0
    successful_pdfs: int = 0
    failed_pdfs: int = 0
    skipped_pdfs: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_stored_vectors: int = 0
    total_processing_time_seconds: float = 0.0

    @property
    def average_processing_time_seconds(self) -> float:
        processed = self.successful_pdfs + self.failed_pdfs
        return self.total_processing_time_seconds / processed if processed else 0.0
