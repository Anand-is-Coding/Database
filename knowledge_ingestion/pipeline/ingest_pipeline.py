"""Ingestion pipeline orchestration.

Wires together the loader, parser, chunker, embedder, and vector store into
a single end-to-end flow: S3 -> Download -> Parse -> Chunk -> Embed ->
Qdrant. This module coordinates only - it does not implement S3, parsing,
chunking, embedding, or Qdrant logic itself, and each of those modules
remains independently usable/testable without this orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TypeVar

from chunking.llama_chunker import LlamaChunker
from config.logging import get_logger
from embedding.bge_embedder import EmbeddingService
from loaders.s3_loader import S3Loader
from models.pipeline import DocumentState, PipelineConfig, PipelineStage, PipelineState, PipelineStatistics
from models.s3_object import S3PdfObject
from parser.docling_parser import DoclingParser
from utils.retry import retry_with_backoff
from vectorstore.qdrant_client import VectorStore

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class _ProgressTracker:
    """Logs a structured one-line progress summary after each document.

    Deliberately log-based rather than a live-redrawing terminal widget:
    this pipeline is meant to run unattended over thousands of PDFs (in a
    container, cron job, or CI log), where a live progress bar's control
    codes get mangled by log aggregation - plain sequential lines with
    elapsed/ETA are more robust there than they'd be nicer in a live TTY.
    """

    total_documents: int
    completed: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def report(self, *, subject: str, document: str, stage: str) -> None:
        elapsed = time.perf_counter() - self.started_at
        remaining = self.total_documents - self.completed
        avg_per_doc = elapsed / self.completed if self.completed else 0.0
        eta = avg_per_doc * remaining
        logger.info(
            "Progress: subject='%s' document='%s' stage='%s' completed=%d/%d remaining=%d "
            "elapsed=%.1fs eta=%.1fs",
            subject,
            document,
            stage,
            self.completed,
            self.total_documents,
            remaining,
            elapsed,
            eta,
        )


class IngestionPipeline:
    """Coordinates S3Loader -> DoclingParser -> LlamaChunker -> EmbeddingService
    -> VectorStore for one PDF at a time, with checkpointed, resumable,
    per-document failure isolation.

    All five collaborators are injectable (dependency injection), so this
    orchestrator can be tested with mocked/stubbed stages instead of the
    real ML models or a real S3/Qdrant connection - and each stage stays
    completely independent of this class and of each other.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        loader: S3Loader | None = None,
        parser: DoclingParser | None = None,
        chunker: LlamaChunker | None = None,
        embedder: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.loader = loader or S3Loader(download_root=self.config.download_root)
        self.parser = parser or DoclingParser()
        self.chunker = chunker or LlamaChunker()
        self.embedder = embedder or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

        self.state = PipelineState.load(self.config.checkpoint_path)
        self.statistics = PipelineStatistics()
        self._unsaved_documents = 0

    # ------------------------------------------------------------------
    # Public API - batch processing: entire bucket, one subject, one PDF
    # ------------------------------------------------------------------

    def run(self) -> PipelineStatistics:
        """Process every subject and every PDF discovered in the bucket."""
        logger.info(
            "Pipeline started: scope=bucket dry_run=%s overwrite_existing=%s",
            self.config.dry_run,
            self.config.overwrite_existing,
        )
        self.statistics = PipelineStatistics()

        subjects = self.loader.discover_subjects()
        pdfs: list[S3PdfObject] = []
        for subject in subjects:
            try:
                pdfs.extend(self.loader.discover_pdf_files(subject))
            except Exception as exc:  # noqa: BLE001 - one subject's discovery failing shouldn't abort the rest
                logger.error("Failed to discover PDFs for subject '%s', skipping subject: %s", subject, exc)

        self._process_documents(pdfs)
        self._log_finished()
        return self.statistics

    def run_subject(self, subject: str) -> PipelineStatistics:
        """Process every PDF discovered under a single subject."""
        logger.info(
            "Pipeline started: scope=subject subject='%s' dry_run=%s overwrite_existing=%s",
            subject,
            self.config.dry_run,
            self.config.overwrite_existing,
        )
        self.statistics = PipelineStatistics()

        pdfs = self.loader.discover_pdf_files(subject)
        self._process_documents(pdfs)
        self._log_finished()
        return self.statistics

    def run_document(self, pdf: S3PdfObject) -> DocumentState:
        """Process a single PDF through the full 7-step workflow, updating
        checkpoint state (and `self.statistics`) as it goes.

        1. Check if already processed (skip if so, unless `overwrite_existing`).
        2. Download if needed.
        3. Parse.
        4. Chunk.
        5. Generate embeddings.
        6. Store vectors.
        7. Mark completed.

        A failure at any stage is caught here - it never propagates out of
        this method - so a caller looping over many documents never needs
        its own try/except to keep going.
        """
        doc_state = self.state.documents.get(pdf.key) or DocumentState(s3_key=pdf.key, subject=pdf.subject)

        # 1. Check if already processed.
        if doc_state.success and not self.config.overwrite_existing:
            logger.info("Skipping already-processed document: '%s' (subject='%s').", pdf.key, pdf.subject)
            self.statistics.skipped_pdfs += 1
            return doc_state

        if self.config.dry_run:
            logger.info("[DRY RUN] Would process: '%s' (subject='%s').", pdf.key, pdf.subject)
            return doc_state

        logger.info("Document started: '%s' (subject='%s').", pdf.key, pdf.subject)
        start_time = time.perf_counter()
        now = self._now()
        doc_state.last_attempted_at = now
        if doc_state.first_attempted_at is None:
            doc_state.first_attempted_at = now
        doc_state.etag = pdf.etag

        try:
            self._run_stages(pdf, doc_state)

            doc_state.stage = PipelineStage.COMPLETED
            doc_state.success = True
            doc_state.last_error = None
            doc_state.completed_at = self._now()
            doc_state.processing_time_seconds = time.perf_counter() - start_time
            self.statistics.successful_pdfs += 1
            self.statistics.total_processing_time_seconds += doc_state.processing_time_seconds
            logger.info("Document completed: '%s' in %.2fs.", pdf.key, doc_state.processing_time_seconds)

        except Exception as exc:  # noqa: BLE001 - a single document's failure must never abort the batch
            doc_state.success = False
            doc_state.last_error = str(exc)
            doc_state.processing_time_seconds = time.perf_counter() - start_time
            self.statistics.failed_pdfs += 1
            self.statistics.total_processing_time_seconds += doc_state.processing_time_seconds
            logger.error(
                "Document failed: '%s' at stage '%s' after %d retr%s: %s",
                pdf.key,
                doc_state.stage.value,
                doc_state.retry_count,
                "y" if doc_state.retry_count == 1 else "ies",
                exc,
            )

        self.state.documents[pdf.key] = doc_state
        self._checkpoint(force=False)
        return doc_state

    def error_report(self) -> list[DocumentState]:
        """Return every document currently recorded as failed, for a final report."""
        return [d for d in self.state.documents.values() if not d.success and d.last_error is not None]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _checkpoint(self, *, force: bool) -> None:
        """Persist `self.state`, honoring `checkpoint_save_interval`.

        In-memory state (`self.state.documents`) is always updated
        immediately regardless of this throttling - only the disk write is
        batched, so `error_report()`/statistics within this same process are
        never stale. `force=True` always writes (used for the final flush at
        the end of a run, so a run never ends with unpersisted progress).
        """
        self._unsaved_documents += 1
        if not force and self._unsaved_documents < self.config.checkpoint_save_interval:
            return
        self.state.save(self.config.checkpoint_path)
        self._unsaved_documents = 0

    def _process_documents(self, pdfs: list[S3PdfObject]) -> None:
        self.statistics.total_pdfs += len(pdfs)
        tracker = _ProgressTracker(total_documents=len(pdfs))

        try:
            for pdf in pdfs:
                doc_state = self.run_document(pdf)
                tracker.completed += 1
                if self.config.show_progress:
                    tracker.report(subject=pdf.subject, document=pdf.filename, stage=doc_state.stage.value)
        finally:
            # Always flush on the way out - including on Ctrl+C/an
            # unexpected exception from discovery/iteration itself - so a
            # run never silently loses progress that was already recorded
            # in memory just because it fell between two scheduled saves.
            self._checkpoint(force=True)

    def _run_stages(self, pdf: S3PdfObject, doc_state: DocumentState) -> None:
        # 2. Download if needed.
        doc_state.stage = PipelineStage.DOWNLOADING
        local_path = self._with_retry(
            lambda: self.loader.download_file(pdf, overwrite=self.config.overwrite_existing), doc_state
        )

        # 3. Parse. subject/class_name are passed through explicitly from the
        # S3Loader's own discovery (S3PdfObject) rather than left for the
        # parser to guess from the local download path — the loader already
        # knows these precisely from the real S3 key, including for bucket
        # layouts with no class-level subfolder (class_name is then "").
        doc_state.stage = PipelineStage.PARSING
        document = self._with_retry(
            lambda: self.parser.parse(local_path, subject=pdf.subject, class_name=pdf.class_name or None),
            doc_state,
        )
        doc_state.document_id = document.metadata.document_id
        logger.info("Parsing complete: '%s' (document_id=%s).", pdf.key, document.metadata.document_id)

        # 4. Chunk.
        doc_state.stage = PipelineStage.CHUNKING
        chunks = self._with_retry(lambda: self.chunker.chunk(document), doc_state)
        self.statistics.total_chunks += len(chunks)
        logger.info("Chunking complete: '%s' (%d chunk(s)).", pdf.key, len(chunks))

        # 5. Generate embeddings.
        doc_state.stage = PipelineStage.EMBEDDING
        embedded_chunks = self._with_retry(lambda: self.embedder.embed_chunks(chunks), doc_state)
        self.statistics.total_embeddings += len(embedded_chunks)
        logger.info("Embedding complete: '%s' (%d vector(s)).", pdf.key, len(embedded_chunks))

        # 6. Store vectors.
        doc_state.stage = PipelineStage.STORING
        upload_results = self._with_retry(lambda: self.vector_store.upsert_vectors(embedded_chunks), doc_state)
        stored = sum(r.uploaded for r in upload_results)
        self.statistics.total_stored_vectors += stored
        logger.info("Upload complete: '%s' (%d vector(s) stored).", pdf.key, stored)

    def _with_retry(self, func: Callable[[], T], doc_state: DocumentState) -> T:
        """Retry any failure at the current stage up to `max_retries` times,
        then give up (the surrounding `try/except` in `run_document` will
        mark the document failed and move on) - a uniform retry-then-skip
        policy rather than classifying transient vs. permanent errors, since
        each stage's own module already retries its own transient failures
        internally before ever raising up to here.
        """

        def _should_retry(_exc: Exception) -> bool:
            # Every failure is retried up to max_attempts, regardless of
            # cause - see docstring above.
            doc_state.retry_count += 1
            return True

        return retry_with_backoff(func, should_retry=_should_retry, max_attempts=self.config.max_retries)

    def _log_finished(self) -> None:
        stats = self.statistics
        logger.info(
            "Pipeline finished: %d total, %d successful, %d failed, %d skipped, %d chunk(s), "
            "%d embedding(s), %d vector(s) stored, avg %.2fs/doc.",
            stats.total_pdfs,
            stats.successful_pdfs,
            stats.failed_pdfs,
            stats.skipped_pdfs,
            stats.total_chunks,
            stats.total_embeddings,
            stats.total_stored_vectors,
            stats.average_processing_time_seconds,
        )
        failures = self.error_report()
        if failures:
            logger.error("Pipeline completed with %d failure(s):", len(failures))
            for failure in failures:
                logger.error("  - '%s' (subject='%s'): %s", failure.s3_key, failure.subject, failure.last_error)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
