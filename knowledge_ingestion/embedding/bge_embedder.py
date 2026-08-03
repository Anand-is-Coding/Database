"""Embedding module — responsible for generating vector embeddings for
document chunks using the BAAI/bge-m3 model.

Single responsibility: `list[Chunk]` -> `list[EmbeddedChunk]`. This module
knows nothing about Qdrant, retrieval, or LLMs — it only loads a
sentence-transformers model once and turns chunk text into normalized,
validated vectors plus a Qdrant-ready payload.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from math import ceil
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.logging import get_logger
from embedding.exceptions import (
    EmbeddingGenerationError,
    EmbeddingTimeoutError,
    EmbeddingValidationError,
    EmptyChunkTextError,
    InvalidVectorError,
    ModelLoadError,
)
from models.chunk import Chunk
from models.embedded_chunk import EmbeddedChunk, EmbeddingConfig
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

_TRANSIENT_LOAD_ERROR_KEYWORDS = (
    "timeout",
    "connection",
    "network",
    "temporarily unavailable",
    "500",
    "502",
    "503",
    "504",
)


def _is_transient_load_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_LOAD_ERROR_KEYWORDS)


def _resolve_device(requested: str | None) -> str:
    """Pick a device, falling back to CPU if the requested one isn't available."""
    if requested is None:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    normalized = requested.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested device '%s' but CUDA is not available; falling back to CPU.", requested)
        return "cpu"
    if normalized == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("Requested device 'mps' but MPS is not available; falling back to CPU.")
        return "cpu"
    return requested


def _model_dimension(model: Any) -> int:
    """Read the embedding dimension off a loaded model, across API name changes."""
    dimension = None
    if hasattr(model, "get_embedding_dimension"):
        dimension = model.get_embedding_dimension()
    if dimension is None and hasattr(model, "get_sentence_embedding_dimension"):
        dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise ModelLoadError("Could not determine the embedding dimension of the loaded model.")
    return dimension


class EmbeddingService:
    """Generates dense vector embeddings for text chunks using BAAI/bge-m3.

    The model is loaded lazily on first use and cached for the lifetime of
    the instance (`Load model once. Do not reload it for every document.`).
    A pre-built model can be injected via the `model` constructor argument
    so tests never need to download the real (multi-GB) model.
    """

    def __init__(self, config: EmbeddingConfig | None = None, model: Any | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = model
        self._dimension: int | None = _model_dimension(model) if model is not None else None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
            self._dimension = _model_dimension(self._model)
        return self._model

    def _load_model(self) -> Any:
        device = _resolve_device(self.config.device)

        def _do_load() -> Any:
            return SentenceTransformer(self.config.model_name, device=device)

        try:
            model = retry_with_backoff(
                _do_load, should_retry=_is_transient_load_error, max_attempts=self.config.max_retries
            )
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise ModelLoadError(
                f"Failed to load embedding model '{self.config.model_name}': {exc}"
            ) from exc

        logger.info("Device selected: %s", device)
        logger.info("Model loaded: '%s'", self.config.model_name)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks, preserving 1:1 correspondence with the input."""
        logger.info("Chunks received: %d", len(chunks))
        if not chunks:
            return []

        empty_ids = [c.chunk_id for c in chunks if not c.text or not c.text.strip()]
        if empty_ids:
            raise EmptyChunkTextError(
                f"{len(empty_ids)} chunk(s) have empty/whitespace-only text: {empty_ids[:5]}"
                + (" ..." if len(empty_ids) > 5 else "")
            )

        dimension = _model_dimension(self.model) if self._dimension is None else self._dimension
        self._dimension = dimension

        vectors, batch_times = self._embed_in_batches([c.text for c in chunks])

        avg_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
        logger.info(
            "Embeddings generated: %d (avg batch time %.3fs across %d batch(es))",
            len(vectors),
            avg_time,
            len(batch_times),
        )

        embedded_chunks = [
            self._build_embedded_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors)
        ]
        self.validate_embedded_chunks(chunks, embedded_chunks)
        return embedded_chunks

    def validate_embedded_chunks(
        self, chunks: list[Chunk], embedded_chunks: list[EmbeddedChunk]
    ) -> None:
        """Verify no missing vectors, correct dimensions, metadata preserved,
        and that embedding count matches chunk count. Raises
        `EmbeddingValidationError` on any violation.
        """
        if len(embedded_chunks) != len(chunks):
            raise EmbeddingValidationError(
                f"Embedding count ({len(embedded_chunks)}) does not match "
                f"chunk count ({len(chunks)})."
            )

        missing_vectors = sum(1 for e in embedded_chunks if not e.vector)
        wrong_dimension = sum(
            1
            for e in embedded_chunks
            if self._dimension is not None and len(e.vector) != self._dimension
        )
        metadata_mismatches = sum(
            1
            for c, e in zip(chunks, embedded_chunks)
            if c.chunk_id != e.chunk_id or c.metadata.document_id != e.document_id
        )

        logger.info(
            "Validation results: %d embedded, %d missing vector(s), %d wrong-dimension, "
            "%d metadata mismatch(es)",
            len(embedded_chunks),
            missing_vectors,
            wrong_dimension,
            metadata_mismatches,
        )

        if missing_vectors or wrong_dimension or metadata_mismatches:
            raise EmbeddingValidationError(
                f"Embedding validation failed: {missing_vectors} missing vector(s), "
                f"{wrong_dimension} wrong-dimension vector(s), "
                f"{metadata_mismatches} metadata mismatch(es)."
            )

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _embed_in_batches(self, texts: list[str]) -> tuple[list[np.ndarray], list[float]]:
        batch_size = self.config.batch_size
        num_batches = ceil(len(texts) / batch_size)
        all_vectors: list[np.ndarray] = []
        batch_times: list[float] = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            logger.info("Processing batch %d/%d (%d chunk(s))", batch_idx + 1, num_batches, len(batch_texts))

            t0 = time.perf_counter()
            batch_vectors = self._encode_batch(batch_texts)
            batch_times.append(time.perf_counter() - t0)
            all_vectors.extend(batch_vectors)

        return all_vectors, batch_times

    def _encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        def _do_encode() -> np.ndarray:
            return self.model.encode(
                texts,
                batch_size=len(texts),
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        try:
            if self.config.encode_timeout_seconds is not None:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_do_encode)
                    try:
                        raw_vectors = future.result(timeout=self.config.encode_timeout_seconds)
                    except FutureTimeoutError as exc:
                        raise EmbeddingTimeoutError(
                            f"Embedding a batch of {len(texts)} chunk(s) exceeded "
                            f"{self.config.encode_timeout_seconds}s."
                        ) from exc
            else:
                raw_vectors = _do_encode()
        except EmbeddingTimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise EmbeddingGenerationError(f"Failed to generate embeddings for batch: {exc}") from exc

        return [self._validate_vector(vector) for vector in raw_vectors]

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        if vector.ndim != 1:
            raise InvalidVectorError(f"Expected a 1-D vector, got shape {vector.shape}.")
        if self._dimension is not None and vector.shape[0] != self._dimension:
            raise InvalidVectorError(
                f"Vector dimension {vector.shape[0]} does not match expected {self._dimension}."
            )
        if not np.isfinite(vector).all():
            raise InvalidVectorError("Vector contains NaN or infinite values.")
        return vector

    # ------------------------------------------------------------------
    # EmbeddedChunk assembly
    # ------------------------------------------------------------------

    def _build_embedded_chunk(self, chunk: Chunk, vector: np.ndarray) -> EmbeddedChunk:
        meta = chunk.metadata
        return EmbeddedChunk(
            vector=vector.tolist(),
            chunk_id=chunk.chunk_id,
            document_id=meta.document_id,
            subject=meta.subject,
            class_name=meta.class_name,
            chapter=meta.chapter,
            section=meta.section,
            page_number=meta.page_number,
            source_pdf=meta.source_pdf,
            chunk_number=meta.chunk_number,
            total_chunks=meta.total_chunks,
            token_count=meta.estimated_token_count,
            character_count=meta.character_count,
            original_text=chunk.text,
            metadata=meta.model_dump(mode="json"),
            embedding_model=self.config.model_name,
        )
