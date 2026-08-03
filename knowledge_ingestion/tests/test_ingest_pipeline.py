"""Unit tests for IngestionPipeline — fake collaborators duck-typing the
real S3Loader/DoclingParser/LlamaChunker/EmbeddingService/VectorStore
interfaces, so orchestration logic (checkpointing, resume, retry-then-skip,
statistics, dry-run, overwrite) is exercised without S3/Docling/torch/Qdrant.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from models.chunk import Chunk, ChunkMetadata
from models.document import Document, Metadata as ParserMetadata, Page
from models.embedded_chunk import EmbeddedChunk
from models.pipeline import PipelineConfig, PipelineStage
from models.qdrant import UploadResult
from models.s3_object import S3PdfObject
from pipeline.ingest_pipeline import IngestionPipeline


def _make_pdf(subject: str, filename: str) -> S3PdfObject:
    return S3PdfObject(
        subject=subject,
        class_name="class11",
        key=f"{subject}/class11/{filename}",
        filename=filename,
        size_bytes=1024,
        last_modified=datetime.now(timezone.utc),
        etag=f"etag-{subject}-{filename}",
    )


class _FakeLoader:
    def __init__(self, catalog: dict[str, list[S3PdfObject]], download_root: Path):
        self.catalog = catalog
        self.download_root = download_root
        self.download_calls = 0

    def discover_subjects(self) -> list[str]:
        return sorted(self.catalog.keys())

    def discover_pdf_files(self, subject: str) -> list[S3PdfObject]:
        return self.catalog.get(subject, [])

    def download_file(self, pdf: S3PdfObject, *, overwrite: bool = False) -> Path:
        self.download_calls += 1
        path = self.download_root / pdf.key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return path


class _SimulatedFailure(Exception):
    pass


class _FakeParser:
    def __init__(self, download_root: Path, fail_keys: set[str] | None = None):
        self.download_root = download_root
        self.fail_keys = fail_keys or set()
        self.parse_calls = 0

    def parse(self, path: Path, *, subject: str | None = None, class_name: str | None = None) -> Document:
        self.parse_calls += 1
        key = str(path.relative_to(self.download_root)).replace("\\", "/")
        if key in self.fail_keys:
            raise _SimulatedFailure(f"simulated permanent parse failure for {key}")
        # Mirrors the real DoclingParser.parse(): prefer the explicit
        # subject/class_name the caller (IngestionPipeline) already knows
        # from S3Loader's discovery, falling back to path-based guessing
        # only when neither was given.
        if subject is None and class_name is None:
            subject = path.parts[-3]
            class_name = path.parts[-2]
        return Document(
            metadata=ParserMetadata(
                document_id=str(uuid.uuid4()),
                subject=subject,
                class_name=class_name or "",
                book_name=path.stem,
                source_path=str(path),
                total_pages=1,
                parser_version="test",
            ),
            title="Fake Title",
            pages=[Page(page_number=1)],
            sections=[],
        )


class _FakeChunker:
    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []
        for i in (1, 2):
            meta = ChunkMetadata(
                document_id=document.metadata.document_id,
                subject=document.metadata.subject,
                class_name=document.metadata.class_name,
                source_pdf=document.metadata.source_path,
                parser_version=document.metadata.parser_version,
                chunk_number=i,
                total_chunks=2,
                character_count=10,
                estimated_token_count=3,
            )
            chunks.append(Chunk(chunk_id=str(uuid.uuid4()), text=f"fake chunk {i}", metadata=meta))
        return chunks


class _FakeEmbedder:
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(
                vector=[0.1, 0.2, 0.3],
                chunk_id=c.chunk_id,
                document_id=c.metadata.document_id,
                subject=c.metadata.subject,
                class_name=c.metadata.class_name,
                source_pdf=c.metadata.source_pdf,
                chunk_number=c.metadata.chunk_number,
                total_chunks=c.metadata.total_chunks,
                token_count=c.metadata.estimated_token_count,
                character_count=c.metadata.character_count,
                original_text=c.text,
                embedding_model="fake-model",
            )
            for c in chunks
        ]


class _FakeVectorStore:
    def __init__(self):
        self.stored: list[EmbeddedChunk] = []

    def upsert_vectors(self, embedded_chunks: list[EmbeddedChunk]) -> list[UploadResult]:
        self.stored.extend(embedded_chunks)
        by_subject: dict[str, int] = {}
        for c in embedded_chunks:
            by_subject[c.subject] = by_subject.get(c.subject, 0) + 1
        return [UploadResult(collection_name=s, requested=n, uploaded=n) for s, n in by_subject.items()]


def _build_pipeline(tmp_path: Path, catalog: dict[str, list[S3PdfObject]], fail_keys: set[str] | None = None, **overrides):
    download_root = tmp_path / "downloads"
    checkpoint_path = tmp_path / "state.json"
    loader = _FakeLoader(catalog, download_root)
    parser = _FakeParser(download_root, fail_keys=fail_keys)
    chunker = _FakeChunker()
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    config = PipelineConfig(
        download_root=download_root, checkpoint_path=checkpoint_path, max_retries=2, show_progress=True, **overrides
    )
    pipeline = IngestionPipeline(
        config=config, loader=loader, parser=parser, chunker=chunker, embedder=embedder, vector_store=vector_store
    )
    return pipeline, loader, parser, chunker, embedder, vector_store


def test_full_run_reports_correct_statistics(tmp_path: Path):
    catalog = {
        "biology": [_make_pdf("biology", "chapter1.pdf"), _make_pdf("biology", "chapter2.pdf")],
        "physics": [_make_pdf("physics", "chapter1.pdf")],
    }
    pipeline, *_, vector_store = _build_pipeline(tmp_path, catalog)

    stats = pipeline.run()

    assert stats.total_pdfs == 3
    assert stats.successful_pdfs == 3
    assert stats.failed_pdfs == 0
    assert stats.total_chunks == 6
    assert stats.total_embeddings == 6
    assert stats.total_stored_vectors == 6
    assert len(vector_store.stored) == 6

    for pdf in catalog["biology"] + catalog["physics"]:
        doc_state = pipeline.state.documents[pdf.key]
        assert doc_state.success is True
        assert doc_state.stage == PipelineStage.COMPLETED
        assert doc_state.document_id is not None


def test_resumed_run_skips_completed_documents_without_reparsing(tmp_path: Path):
    catalog = {"biology": [_make_pdf("biology", "chapter1.pdf"), _make_pdf("biology", "chapter2.pdf")]}
    pipeline1, *_ = _build_pipeline(tmp_path, catalog)
    pipeline1.run()

    assert pipeline1.config.checkpoint_path.exists()

    # A brand-new IngestionPipeline instance, same checkpoint path — simulates
    # a fresh process after a crash/restart.
    pipeline2, _, parser2, *_ = _build_pipeline(tmp_path, catalog)
    stats2 = pipeline2.run()

    assert stats2.successful_pdfs == 0
    assert stats2.skipped_pdfs == 2
    assert parser2.parse_calls == 0


def test_one_failing_document_does_not_block_the_others(tmp_path: Path):
    catalog = {
        "biology": [
            _make_pdf("biology", "good1.pdf"),
            _make_pdf("biology", "corrupted.pdf"),
            _make_pdf("biology", "good2.pdf"),
        ]
    }
    fail_key = "biology/class11/corrupted.pdf"
    pipeline, *_ = _build_pipeline(tmp_path, catalog, fail_keys={fail_key})

    stats = pipeline.run()

    assert stats.successful_pdfs == 2
    assert stats.failed_pdfs == 1

    failed_state = pipeline.state.documents[fail_key]
    assert failed_state.success is False
    assert failed_state.stage == PipelineStage.PARSING
    assert "simulated permanent parse failure" in failed_state.last_error

    report = pipeline.error_report()
    assert len(report) == 1 and report[0].s3_key == fail_key


def test_overwrite_existing_forces_reprocessing(tmp_path: Path):
    catalog = {"biology": [_make_pdf("biology", "chapter1.pdf")]}
    pipeline1, _, parser1, *_ = _build_pipeline(tmp_path, catalog)
    pipeline1.run()
    assert parser1.parse_calls == 1

    pipeline2, _, parser2, *_ = _build_pipeline(tmp_path, catalog, overwrite_existing=True)
    stats2 = pipeline2.run()

    assert stats2.successful_pdfs == 1
    assert parser2.parse_calls == 1


def test_dry_run_touches_no_stage(tmp_path: Path):
    catalog = {"biology": [_make_pdf("biology", "chapter1.pdf")]}
    pipeline, loader, parser, chunker, embedder, vector_store = _build_pipeline(tmp_path, catalog, dry_run=True)

    stats = pipeline.run()

    assert loader.download_calls == 0
    assert parser.parse_calls == 0
    assert len(vector_store.stored) == 0
    assert stats.successful_pdfs == 0


def test_run_subject_only_processes_that_subject(tmp_path: Path):
    catalog = {
        "biology": [_make_pdf("biology", "chapter1.pdf")],
        "physics": [_make_pdf("physics", "chapter1.pdf")],
    }
    pipeline, *_ = _build_pipeline(tmp_path, catalog)

    stats = pipeline.run_subject("biology")

    assert stats.total_pdfs == 1
    assert "physics/class11/chapter1.pdf" not in pipeline.state.documents


def test_run_document_processes_a_single_pdf_directly(tmp_path: Path):
    catalog = {"biology": [_make_pdf("biology", "chapter1.pdf")]}
    pipeline, *_ = _build_pipeline(tmp_path, catalog)

    doc_state = pipeline.run_document(catalog["biology"][0])

    assert doc_state.success is True
    assert pipeline.statistics.successful_pdfs == 1


def test_checkpoint_file_round_trips_via_pipeline_state_load(tmp_path: Path):
    catalog = {"biology": [_make_pdf("biology", "chapter1.pdf")]}
    pipeline, *_ = _build_pipeline(tmp_path, catalog)
    pipeline.run()

    from models.pipeline import PipelineState

    reloaded = PipelineState.load(pipeline.config.checkpoint_path)
    key = catalog["biology"][0].key
    assert reloaded.documents[key].success is True


def test_checkpoint_save_interval_reduces_disk_writes_but_final_flush_always_happens(tmp_path: Path, monkeypatch):
    from models.pipeline import PipelineState

    catalog = {"biology": [_make_pdf("biology", f"chapter{i}.pdf") for i in range(1, 5)]}
    pipeline, *_ = _build_pipeline(tmp_path, catalog, checkpoint_save_interval=10)

    save_calls = []
    original_save = PipelineState.save

    def counting_save(self, path):
        save_calls.append(path)
        return original_save(self, path)

    # PipelineState is a Pydantic model — instance-level monkeypatching of a
    # method is rejected by its __setattr__, so patch the class instead.
    monkeypatch.setattr(PipelineState, "save", counting_save)

    pipeline.run()

    # 4 documents with interval=10 should trigger exactly one save — the
    # final forced flush at the end of the run — not 4.
    assert len(save_calls) == 1
    assert pipeline.config.checkpoint_path.exists()
    for pdf in catalog["biology"]:
        assert pipeline.state.documents[pdf.key].success is True
