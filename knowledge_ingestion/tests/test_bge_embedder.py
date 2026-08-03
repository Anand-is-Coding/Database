"""Unit tests for EmbeddingService — a fake injected model (fast, hermetic)
covers validation/error paths; one @integration test exercises a real
(small) sentence-transformers model end to end.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from embedding.bge_embedder import EmbeddingService
from embedding.exceptions import (
    EmbeddingTimeoutError,
    EmptyChunkTextError,
    InvalidVectorError,
    ModelLoadError,
)
from models.chunk import Chunk, ChunkMetadata
from models.embedded_chunk import EmbeddingConfig


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        document_id="doc-123",
        subject="biology",
        class_name="class11",
        chapter="Chapter 1: The Cell",
        section="1.1 Cell Types",
        page_number=1,
        page_numbers=[1],
        source_pdf="downloads/biology/class11/chapter1.pdf",
        parser_version="test",
        chunk_number=1,
        total_chunks=1,
        character_count=len(text),
        estimated_token_count=len(text.split()),
    )
    return Chunk(chunk_id=chunk_id, text=text, metadata=meta)


class _FakeModel:
    def __init__(self, dim: int = 8, vectors=None, delay: float = 0.0, raise_on_encode: Exception | None = None):
        self.dim = dim
        self.vectors = vectors
        self.delay = delay
        self.raise_on_encode = raise_on_encode

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        if self.delay:
            time.sleep(self.delay)
        if self.raise_on_encode:
            raise self.raise_on_encode
        if self.vectors is not None:
            return np.array(self.vectors)
        return np.random.RandomState(0).rand(len(texts), self.dim).astype(np.float32)


def test_empty_chunk_text_rejected():
    service = EmbeddingService(config=EmbeddingConfig(), model=_FakeModel())
    chunks = [_make_chunk("c1", "Valid text."), _make_chunk("c2", "   ")]

    with pytest.raises(EmptyChunkTextError) as exc_info:
        service.embed_chunks(chunks)
    assert "c2" in str(exc_info.value)


def test_invalid_vector_dimension_rejected():
    fake = _FakeModel(dim=8, vectors=[[1.0] * 5])
    service = EmbeddingService(config=EmbeddingConfig(), model=fake)

    with pytest.raises(InvalidVectorError):
        service.embed_chunks([_make_chunk("c1", "Some text.")])


def test_invalid_vector_nan_rejected():
    fake = _FakeModel(dim=4, vectors=[[float("nan")] * 4])
    service = EmbeddingService(config=EmbeddingConfig(), model=fake)

    with pytest.raises(InvalidVectorError):
        service.embed_chunks([_make_chunk("c1", "Some text.")])


def test_batch_timeout_raises():
    fake = _FakeModel(dim=4, delay=1.0)
    service = EmbeddingService(config=EmbeddingConfig(encode_timeout_seconds=0.1), model=fake)

    with pytest.raises(EmbeddingTimeoutError):
        service.embed_chunks([_make_chunk("c1", "Some text.")])


def test_model_load_failure_raises_model_load_error():
    service = EmbeddingService(
        config=EmbeddingConfig(model_name="this-model-definitely-does-not-exist-xyz", max_retries=1)
    )
    with pytest.raises(ModelLoadError):
        service.embed_chunks([_make_chunk("c1", "Some text.")])


def test_embed_chunks_preserves_metadata_and_order():
    fake = _FakeModel(dim=4)
    service = EmbeddingService(config=EmbeddingConfig(), model=fake)
    chunks = [_make_chunk("c1", "first"), _make_chunk("c2", "second")]

    embedded = service.embed_chunks(chunks)

    assert len(embedded) == 2
    assert [e.chunk_id for e in embedded] == ["c1", "c2"]
    assert all(e.document_id == "doc-123" for e in embedded)
    assert all(e.subject == "biology" for e in embedded)
    assert all(len(e.vector) == 4 for e in embedded)


@pytest.mark.integration
def test_real_small_model_produces_normalized_vectors():
    chunks = [
        _make_chunk("c1", "The cell is the basic unit of life."),
        _make_chunk("c2", "Prokaryotic cells lack a nucleus."),
    ]
    service = EmbeddingService(
        config=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=2)
    )
    embedded = service.embed_chunks(chunks)

    assert len(embedded) == 2
    assert len({len(e.vector) for e in embedded}) == 1
    for e in embedded:
        norm = np.linalg.norm(e.vector)
        assert abs(norm - 1.0) < 1e-4
