"""Vector store module — responsible for persisting embedded chunks in Qdrant.

Single responsibility: `list[EmbeddedChunk]` -> vectors durably stored in
Qdrant, one collection per subject. This module knows nothing about
semantic search, retrieval, or LLMs — it only writes.

Collections are never hardcoded: the collection a chunk lands in is always
derived from that chunk's own `subject` field, so a brand-new subject
uploaded to S3 tomorrow gets its own collection automatically the first
time a chunk for it is upserted.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config.logging import get_logger
from models.embedded_chunk import EmbeddedChunk
from models.qdrant import CollectionStats, QdrantConfig, UploadResult
from utils.qdrant_errors import is_transient_qdrant_error
from utils.retry import retry_with_backoff
from vectorstore.exceptions import (
    BatchUploadError,
    CollectionCreationError,
    InvalidPayloadError,
    QdrantAuthenticationError,
    QdrantConnectionError,
    VectorStoreError,
)

logger = get_logger(__name__)

_AUTH_HTTP_STATUSES = {401, 403}

# Payload fields worth a dedicated index by default: `document_id` is used
# by `delete_document`'s filter on every call, and `class`/`chapter` are the
# natural within-subject filters a tutor would apply. `subject` itself is
# deliberately not indexed here — it's already implied by which collection
# is being queried, so indexing it again would be redundant.
_INDEX_FIELD_TYPES: dict[str, qmodels.PayloadSchemaType] = {
    "document_id": qmodels.PayloadSchemaType.KEYWORD,
    "class": qmodels.PayloadSchemaType.KEYWORD,
    "chapter": qmodels.PayloadSchemaType.KEYWORD,
    "section": qmodels.PayloadSchemaType.KEYWORD,
    "page": qmodels.PayloadSchemaType.INTEGER,
}


def _map_qdrant_error(exc: Exception, *, action: str) -> VectorStoreError:
    if isinstance(exc, UnexpectedResponse) and exc.status_code in _AUTH_HTTP_STATUSES:
        return QdrantAuthenticationError(f"Qdrant rejected credentials while trying to {action}.")
    if isinstance(exc, (ResponseHandlingException, UnexpectedResponse)):
        return QdrantConnectionError(f"Failed to {action}: {exc}")
    return QdrantConnectionError(f"Unexpected error while trying to {action}: {exc}")


class CollectionManager:
    """Owns Qdrant collection lifecycle: existence, creation, stats, deletion.

    Takes an already-connected `QdrantClient` (dependency injection) so it
    never manages its own connection — that's `VectorStore.connect()`'s job.
    """

    def __init__(self, client: QdrantClient, config: QdrantConfig) -> None:
        self._client = client
        self.config = config
        # Once a collection is confirmed to exist, it stays "known" for the
        # lifetime of this instance - `collection_exists`/`create_collection`
        # are called on every single upsert/search (once per document during
        # ingestion, once per query during retrieval), and the overwhelming
        # majority of those calls are "yes, still exists" after the first
        # one. Without this cache, a 100,000-document ingestion run makes
        # ~100,000 redundant existence-check round-trips to Qdrant for
        # collections that were confirmed to exist on document #1.
        self._known_collections: set[str] = set()

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable and responding."""
        try:
            self._call_with_retry(self._client.get_collections, action="check Qdrant health")
            return True
        except VectorStoreError as exc:
            logger.error("Health check failed: %s", exc)
            return False

    def collection_exists(self, collection_name: str) -> bool:
        if collection_name in self._known_collections:
            return True
        exists = self._call_with_retry(
            lambda: self._client.collection_exists(collection_name),
            action=f"check existence of collection '{collection_name}'",
        )
        if exists:
            self._known_collections.add(collection_name)
        return exists

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        """Create a collection with cosine distance if it doesn't already exist.

        Idempotent: if the collection already exists, this is a no-op that
        logs a "reused" message rather than raising.
        """
        if self.collection_exists(collection_name):
            logger.info("Collection selected: '%s' (already exists, reused).", collection_name)
            return False

        try:
            self._call_with_retry(
                lambda: self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    ),
                ),
                action=f"create collection '{collection_name}'",
            )
        except VectorStoreError as exc:
            raise CollectionCreationError(
                f"Failed to create collection '{collection_name}' (size={vector_size}): {exc}"
            ) from exc

        logger.info("Collection created: '%s' (size=%d, distance=cosine).", collection_name, vector_size)
        self._known_collections.add(collection_name)
        self.ensure_indexes(collection_name)
        return True

    def ensure_indexes(self, collection_name: str, fields: list[str] | None = None) -> None:
        """Create payload indexes for fields worth filtering on."""
        for field_name in fields or self.config.indexed_payload_fields:
            field_schema = _INDEX_FIELD_TYPES.get(field_name, qmodels.PayloadSchemaType.KEYWORD)
            try:
                self._call_with_retry(
                    lambda fn=field_name, fs=field_schema: self._client.create_payload_index(
                        collection_name=collection_name, field_name=fn, field_schema=fs
                    ),
                    action=f"create payload index '{field_name}' on '{collection_name}'",
                )
            except VectorStoreError as exc:
                # Indexes are an optimization, not a correctness requirement -
                # a failure here shouldn't block ingestion.
                logger.warning("Could not create payload index '%s' on '%s': %s", field_name, collection_name, exc)

    def get_collection(self, collection_name: str) -> qmodels.CollectionInfo:
        return self._call_with_retry(
            lambda: self._client.get_collection(collection_name),
            action=f"get collection '{collection_name}'",
        )

    def list_collections(self) -> list[str]:
        response = self._call_with_retry(self._client.get_collections, action="list collections")
        return [c.name for c in response.collections]

    def count_vectors(self, collection_name: str) -> int:
        result = self._call_with_retry(
            lambda: self._client.count(collection_name=collection_name, exact=True),
            action=f"count vectors in '{collection_name}'",
        )
        return result.count

    def collection_stats(self, collection_name: str) -> CollectionStats:
        info = self.get_collection(collection_name)
        return CollectionStats(
            name=collection_name,
            points_count=info.points_count or 0,
            indexed_vectors_count=info.indexed_vectors_count,
            segments_count=info.segments_count,
            status=info.status.value,
        )

    def delete_points_by_field(self, collection_name: str, field: str, value: str) -> int:
        """Delete every point where `payload[field] == value`. Returns the
        count of matching points *before* deletion (Qdrant's delete-by-filter
        doesn't report how many it removed).
        """
        count_before = self.count_vectors_matching(collection_name, field, value)
        if count_before == 0:
            return 0

        self._call_with_retry(
            lambda: self._client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))]
                    )
                ),
            ),
            action=f"delete points where {field}='{value}' from '{collection_name}'",
        )
        return count_before

    def count_vectors_matching(self, collection_name: str, field: str, value: str) -> int:
        result = self._call_with_retry(
            lambda: self._client.count(
                collection_name=collection_name,
                count_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))]
                ),
                exact=True,
            ),
            action=f"count points where {field}='{value}' in '{collection_name}'",
        )
        return result.count

    def delete_collection(self, collection_name: str) -> bool:
        if not self.collection_exists(collection_name):
            logger.warning("Collection '%s' does not exist; nothing to delete.", collection_name)
            return False
        self._call_with_retry(
            lambda: self._client.delete_collection(collection_name), action=f"delete collection '{collection_name}'"
        )
        self._known_collections.discard(collection_name)
        logger.info("Collection deleted: '%s'.", collection_name)
        return True

    def _call_with_retry(self, func: Any, *, action: str) -> Any:
        try:
            return retry_with_backoff(
                func, should_retry=is_transient_qdrant_error, max_attempts=self.config.max_retries
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed VectorStoreError below
            raise _map_qdrant_error(exc, action=action) from exc


class BatchUploader:
    """Owns chunked point construction and upload mechanics for one collection.

    Takes an already-connected `QdrantClient` (dependency injection) — it
    has no collection-lifecycle responsibilities of its own.
    """

    def __init__(self, client: QdrantClient, config: QdrantConfig) -> None:
        self._client = client
        self.config = config

    def upload(self, collection_name: str, chunks: list[EmbeddedChunk]) -> UploadResult:
        points, skipped = self._build_points(chunks)
        batch_size = self.config.batch_size
        num_batches = -(-len(points) // batch_size) if points else 0  # ceil div
        uploaded = 0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            batch = points[start : start + batch_size]

            def _do_upsert(b: list[qmodels.PointStruct] = batch) -> qmodels.UpdateResult:
                return self._client.upsert(collection_name=collection_name, points=b, wait=True)

            try:
                retry_with_backoff(
                    _do_upsert, should_retry=is_transient_qdrant_error, max_attempts=self.config.max_retries
                )
            except Exception as exc:  # noqa: BLE001 - wrapped into a typed error below
                raise BatchUploadError(
                    f"Failed to upload batch {batch_idx + 1}/{num_batches} "
                    f"({len(batch)} point(s)) to '{collection_name}': {exc}"
                ) from exc

            uploaded += len(batch)
            logger.info(
                "Batch uploaded: %d/%d (%d point(s)) -> '%s'.",
                batch_idx + 1,
                num_batches,
                len(batch),
                collection_name,
            )

        return UploadResult(
            collection_name=collection_name,
            requested=len(chunks),
            uploaded=uploaded,
            skipped_invalid=skipped,
            batches=num_batches,
        )

    def _build_points(self, chunks: list[EmbeddedChunk]) -> tuple[list[qmodels.PointStruct], int]:
        points: list[qmodels.PointStruct] = []
        skipped = 0
        for chunk in chunks:
            try:
                points.append(self._build_point(chunk))
            except InvalidPayloadError as exc:
                skipped += 1
                logger.warning("Skipping chunk '%s': %s", chunk.chunk_id, exc)
        return points, skipped

    @staticmethod
    def _build_point(chunk: EmbeddedChunk) -> qmodels.PointStruct:
        # The point ID IS the chunk_id (a UUID minted once, in the chunking
        # stage) - this is what makes re-uploading the same chunk an update
        # rather than a duplicate. Validated here rather than trusted blindly,
        # since this module must stay independently robust.
        try:
            uuid.UUID(chunk.chunk_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise InvalidPayloadError(
                f"chunk_id '{chunk.chunk_id}' is not a valid UUID and cannot be used as a Qdrant point ID."
            ) from exc

        if not chunk.vector:
            raise InvalidPayloadError(f"Chunk '{chunk.chunk_id}' has an empty vector.")
        if not chunk.document_id or not chunk.source_pdf:
            raise InvalidPayloadError(f"Chunk '{chunk.chunk_id}' is missing document_id/source_pdf.")

        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "subject": chunk.subject,
            "class": chunk.class_name,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "page": chunk.page_number,
            "source_pdf": chunk.source_pdf,
            "chunk_number": chunk.chunk_number,
            "total_chunks": chunk.total_chunks,
            "token_count": chunk.token_count,
            "embedding_model": chunk.embedding_model,
            "text": chunk.original_text,
            "metadata": chunk.metadata,
            # Not in the spec's example payload, but present on EmbeddedChunk
            # and covered by "Store ALL metadata" - dropping them would be a
            # silent metadata loss.
            "character_count": chunk.character_count,
            "embedding_timestamp": chunk.embedding_timestamp.isoformat(),
        }

        return qmodels.PointStruct(id=chunk.chunk_id, vector=chunk.vector, payload=payload)


class VectorStore:
    """Facade over `CollectionManager` + `BatchUploader` - the single entry
    point a caller (a future pipeline) actually needs.

    `client` can be injected for testing (a mocked/stubbed `QdrantClient`)
    instead of talking to a real Qdrant instance; otherwise one is built
    lazily from `QdrantConfig` on first use.
    """

    def __init__(self, config: QdrantConfig | None = None, client: QdrantClient | None = None) -> None:
        self.config = config or QdrantConfig()
        self._client = client
        self._collections: CollectionManager | None = None
        self._uploader: BatchUploader | None = None
        if client is not None:
            self._collections = CollectionManager(client, self.config)
            self._uploader = BatchUploader(client, self.config)

    def connect(self) -> QdrantClient:
        """Establish (or return the cached) connection to Qdrant. Idempotent."""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=int(self.config.timeout_seconds),
                )
            except Exception as exc:  # noqa: BLE001 - backend raises various types
                raise QdrantConnectionError(f"Failed to connect to Qdrant at '{self.config.url}': {exc}") from exc

            self._collections = CollectionManager(self._client, self.config)
            self._uploader = BatchUploader(self._client, self.config)

            if not self._collections.health_check():
                raise QdrantConnectionError(f"Connected to '{self.config.url}' but health check failed.")
            logger.info("Connected to Qdrant: %s", self.config.url)

        return self._client

    @property
    def collections(self) -> CollectionManager:
        self.connect()
        assert self._collections is not None
        return self._collections

    @property
    def uploader(self) -> BatchUploader:
        self.connect()
        assert self._uploader is not None
        return self._uploader

    # ------------------------------------------------------------------
    # Collection lifecycle (delegates to CollectionManager)
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        return self.collections.health_check()

    def collection_exists(self, collection_name: str) -> bool:
        return self.collections.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        return self.collections.create_collection(collection_name, vector_size)

    def get_collection(self, collection_name: str) -> qmodels.CollectionInfo:
        return self.collections.get_collection(collection_name)

    def list_collections(self) -> list[str]:
        return self.collections.list_collections()

    def count_vectors(self, collection_name: str) -> int:
        return self.collections.count_vectors(collection_name)

    def collection_stats(self, collection_name: str) -> CollectionStats:
        return self.collections.collection_stats(collection_name)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upsert_vectors(self, chunks: list[EmbeddedChunk]) -> list[UploadResult]:
        """Upsert chunks into Qdrant, one collection per subject.

        Chunks are partitioned by `subject` (falling back to
        `QdrantConfig.fallback_collection` for the rare chunk with no
        subject) so a batch spanning multiple subjects lands in the right
        collection for each. A collection is created automatically the
        first time a subject is seen, sized from that batch's own vectors -
        never hardcoded.
        """
        if not chunks:
            return []

        by_collection: dict[str, list[EmbeddedChunk]] = {}
        for chunk in chunks:
            collection_name = chunk.subject or self.config.fallback_collection
            if not collection_name:
                raise InvalidPayloadError(
                    f"Chunk '{chunk.chunk_id}' has no subject and no fallback_collection is configured."
                )
            by_collection.setdefault(collection_name, []).append(chunk)

        results: list[UploadResult] = []
        for collection_name, group in by_collection.items():
            vector_size = self.config.vector_size or len(group[0].vector)
            self.collections.create_collection(collection_name, vector_size)

            result = self.uploader.upload(collection_name, group)
            results.append(result)
            logger.info(
                "Upload completed: '%s' -> %d/%d point(s) uploaded, %d skipped.",
                collection_name,
                result.uploaded,
                result.requested,
                result.skipped_invalid,
            )

        return results

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, subject: str, document_id: str) -> int:
        """Delete every point belonging to one document within a subject's collection."""
        if not self.collections.collection_exists(subject):
            logger.warning("Collection '%s' does not exist; nothing to delete for document '%s'.", subject, document_id)
            return 0
        deleted = self.collections.delete_points_by_field(subject, "document_id", document_id)
        logger.info("Deleted %d point(s) for document '%s' from '%s'.", deleted, document_id, subject)
        return deleted

    def delete_subject(self, subject: str) -> bool:
        """Delete an entire subject's collection (every document within it)."""
        return self.collections.delete_collection(subject)
