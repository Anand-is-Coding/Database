"""Retrieval module — searches Qdrant for the chunks most relevant to a
student's question, given the AI Teacher's current teaching state.

Single responsibility: `TeachingContext` -> `list[RetrievedChunk]`. This
module knows nothing about an LLM, tutor orchestration, voice, animation,
APIs, or a frontend — it only searches and ranks. It reuses the already-
built `EmbeddingService` (to embed the question with the same model used to
embed the corpus) and `VectorStore` (to discover/connect to collections),
but never reaches into a retriever, LLM, or tutor of its own.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from qdrant_client import models as qmodels

from config.logging import get_logger
from embedding.bge_embedder import EmbeddingService
from models.retrieval import RetrievalConfig, RetrievedChunk, TeachingContext
from retrieval.exceptions import (
    CollectionNotFoundError,
    QueryEmbeddingError,
    RetrievalValidationError,
    SearchError,
    SearchTimeoutError,
)
from utils.qdrant_errors import is_transient_qdrant_error
from utils.retry import retry_with_backoff
from vectorstore.qdrant_client import VectorStore

logger = get_logger(__name__)


class Retriever:
    """Turns a `TeachingContext` into ranked, relevant `RetrievedChunk`s.

    Composes an `EmbeddingService` (to embed the question) and a
    `VectorStore` (to discover collections and reach the underlying Qdrant
    connection) - both injectable, so tests never need to load the real
    embedding model or talk to a real Qdrant server.
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._vector_store = vector_store or VectorStore()
        self._embedding_service = embedding_service or EmbeddingService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, context: TeachingContext) -> list[RetrievedChunk]:
        """The main flow: TeachingContext -> collection selection -> search -> ranked chunks.

        If `context.subject` is known, only that subject's collection is
        searched (falling back to an empty result set, not an exception, if
        that subject has no ingested content yet). If `subject` is unknown,
        every discovered collection is searched and results are merged.
        """
        logger.info(
            "Question received: %r (subject=%s, class=%s, chapter=%s, topic=%s, page=%s)",
            context.student_question,
            context.subject,
            context.class_name,
            context.chapter,
            context.current_topic,
            context.current_page,
        )
        if not context.student_question or not context.student_question.strip():
            raise QueryEmbeddingError("TeachingContext.student_question must not be empty.")

        if context.subject:
            try:
                chunks = self.retrieve_by_subject(context)
            except CollectionNotFoundError as exc:
                logger.warning("%s Returning an empty result set.", exc)
                chunks = []
        else:
            chunks = self.retrieve_all(context)

        self.validate_results(chunks)
        return chunks

    def retrieve_by_subject(self, context: TeachingContext) -> list[RetrievedChunk]:
        """Search only `context.subject`'s collection. Raises `CollectionNotFoundError`
        if that subject has no collection yet - unlike `retrieve()`, this method
        is explicit about the subject the caller wants, so a missing
        collection is surfaced rather than silently swallowed.
        """
        if not context.subject:
            raise ValueError("retrieve_by_subject requires context.subject to be set.")
        return self._search_collection(context.subject, context)

    def retrieve_by_chapter(self, context: TeachingContext) -> list[RetrievedChunk]:
        """Search `context.subject`'s collection, filtered to `context.chapter`."""
        if not context.subject or not context.chapter:
            raise ValueError("retrieve_by_chapter requires both context.subject and context.chapter.")
        return self.retrieve_by_subject(context)

    def retrieve_all(self, context: TeachingContext) -> list[RetrievedChunk]:
        """Search every discovered collection (used when `context.subject` is unknown).

        A single collection failing to search is logged and skipped rather
        than aborting the whole cross-subject search.
        """
        collection_names = self._vector_store.list_collections()
        if not collection_names:
            logger.warning("No collections exist yet; returning no results.")
            return []

        all_chunks: list[RetrievedChunk] = []
        for collection_name in collection_names:
            try:
                all_chunks.extend(self._search_collection(collection_name, context, apply_top_k=False))
            except SearchError as exc:
                logger.warning("Skipping collection '%s' due to search failure: %s", collection_name, exc)

        ranked = self.rank_results(all_chunks)
        logger.info(
            "Collections searched: %s -> %d chunk(s) after merging/ranking.",
            collection_names,
            len(ranked),
        )
        return ranked

    # ------------------------------------------------------------------
    # Filtering / ranking
    # ------------------------------------------------------------------

    def build_filters(self, context: TeachingContext) -> qmodels.Filter | None:
        """Build a Qdrant filter from whichever context/config fields are set.

        `subject` is deliberately excluded - it's expressed by which
        collection is searched, not a payload condition within it.
        `board`/`language` are supported per the spec, but note: the
        ingestion payload built in `vectorstore/qdrant_client.py` does not
        currently store either field, so these filters are forward-looking
        until an ingestion run populates them - see the README.
        """
        must: list[qmodels.FieldCondition] = []

        if context.class_name:
            must.append(qmodels.FieldCondition(key="class", match=qmodels.MatchValue(value=context.class_name)))
        if context.chapter:
            must.append(qmodels.FieldCondition(key="chapter", match=qmodels.MatchValue(value=context.chapter)))
        if context.board:
            must.append(qmodels.FieldCondition(key="board", match=qmodels.MatchValue(value=context.board)))
        if context.language:
            must.append(qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=context.language)))
        if self.config.section:
            must.append(qmodels.FieldCondition(key="section", match=qmodels.MatchValue(value=self.config.section)))
        if self.config.page_min is not None or self.config.page_max is not None:
            must.append(
                qmodels.FieldCondition(
                    key="page",
                    range=qmodels.Range(gte=self.config.page_min, lte=self.config.page_max),
                )
            )

        if not must:
            return None
        logger.info("Filters applied: %s", [c.key for c in must])
        return qmodels.Filter(must=must)

    def rank_results(self, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        """Deduplicate by `chunk_id` (keeping the highest score), sort by score
        descending, and truncate to `top_k` (defaults to `self.config.top_k`).
        """
        limit = top_k if top_k is not None else self.config.top_k
        best_by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best_by_id[chunk.chunk_id] = chunk
        ranked = sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)
        return ranked[:limit]

    def validate_results(self, chunks: list[RetrievedChunk]) -> None:
        """Verify no duplicate chunks, every chunk has a score, and metadata
        survived the round trip. Raises `RetrievalValidationError` only for
        genuine data-integrity problems (missing chunk_id/document_id);
        duplicates are logged, not fatal, since `rank_results` already
        dedupes before this runs and this is a defense-in-depth check.
        """
        chunk_ids = [c.chunk_id for c in chunks]
        duplicates = len(chunk_ids) - len(set(chunk_ids))
        missing_metadata = sum(1 for c in chunks if not c.chunk_id or not c.document_id)
        missing_scores = sum(1 for c in chunks if c.score is None)

        logger.info(
            "Validation results: %d chunk(s), %d duplicate(s), %d missing score(s), %d missing metadata",
            len(chunks),
            duplicates,
            missing_scores,
            missing_metadata,
        )
        if duplicates:
            logger.warning("%d duplicate chunk_id(s) found after ranking - unexpected.", duplicates)

        if missing_metadata or missing_scores:
            raise RetrievalValidationError(
                f"Retrieval validation failed: {missing_scores} missing score(s), "
                f"{missing_metadata} chunk(s) missing chunk_id/document_id."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _search_collection(
        self, collection_name: str, context: TeachingContext, *, apply_top_k: bool = True
    ) -> list[RetrievedChunk]:
        if not self._vector_store.collection_exists(collection_name):
            raise CollectionNotFoundError(f"No collection found for subject '{collection_name}'.")

        query_vector = self._embed_query(context)
        qdrant_filter = self.build_filters(context)

        t0 = time.perf_counter()
        response = self._search_with_retry(collection_name, query_vector, qdrant_filter)
        latency = time.perf_counter() - t0

        chunks = [self._to_retrieved_chunk(point) for point in response.points]
        if apply_top_k:
            chunks = self.rank_results(chunks)

        logger.info(
            "Collection searched: '%s' -> %d chunk(s) in %.3fs.", collection_name, len(chunks), latency
        )
        return chunks

    def _embed_query(self, context: TeachingContext) -> list[float]:
        query_text = self._build_query_text(context)
        try:
            vector = self._embedding_service.model.encode(
                [query_text],
                normalize_embeddings=self._embedding_service.config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise QueryEmbeddingError(f"Failed to embed student question: {exc}") from exc
        return vector.tolist()

    @staticmethod
    def _build_query_text(context: TeachingContext) -> str:
        # Blending in the current topic gives the embedding real semantic
        # signal for short/ambiguous questions ("why does it happen?" is
        # meaningless alone, but paired with the topic it isn't).
        if context.current_topic:
            return f"Topic: {context.current_topic}\nQuestion: {context.student_question}"
        return context.student_question

    def _search_with_retry(
        self, collection_name: str, query_vector: list[float], qdrant_filter: qmodels.Filter | None
    ) -> qmodels.QueryResponse:
        search_params = qmodels.SearchParams(**self.config.search_params) if self.config.search_params else None
        client = self._vector_store.connect()

        def _do_search() -> qmodels.QueryResponse:
            return client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=self.config.top_k,
                score_threshold=self.config.score_threshold,
                search_params=search_params,
                with_payload=True,
            )

        def _do_search_with_retry() -> qmodels.QueryResponse:
            return retry_with_backoff(
                _do_search, should_retry=is_transient_qdrant_error, max_attempts=self.config.max_retries
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_search_with_retry)
                try:
                    return future.result(timeout=self.config.timeout_seconds)
                except FutureTimeoutError as exc:
                    raise SearchTimeoutError(
                        f"Search on '{collection_name}' exceeded {self.config.timeout_seconds}s."
                    ) from exc
        except SearchTimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            raise SearchError(f"Search failed on collection '{collection_name}': {exc}") from exc

    @staticmethod
    def _to_retrieved_chunk(point: qmodels.ScoredPoint) -> RetrievedChunk:
        payload: dict[str, Any] = point.payload or {}
        return RetrievedChunk(
            chunk_id=payload.get("chunk_id", str(point.id)),
            document_id=payload.get("document_id", ""),
            text=payload.get("text", ""),
            score=point.score,
            subject=payload.get("subject"),
            class_name=payload.get("class"),
            chapter=payload.get("chapter"),
            section=payload.get("section"),
            page_number=payload.get("page"),
            source_pdf=payload.get("source_pdf"),
            metadata=payload.get("metadata", {}),
        )
