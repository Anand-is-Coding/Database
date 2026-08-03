"""Unit tests for VectorStore — qdrant-client's real in-memory mode
(`QdrantClient(":memory:")`) is fast and fully hermetic (no network, no
external server), so these run as plain unit tests, not @integration.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from models.embedded_chunk import EmbeddedChunk
from models.qdrant import QdrantConfig
from vectorstore.qdrant_client import VectorStore


def _make_chunk(subject="biology", document_id="doc-1", text="The cell is the basic unit of life.",
                 chunk_id=None, dim=8, seed=0.1, chapter="Chapter 1: The Cell", class_name="class11", page=1):
    vec = np.random.RandomState(int(seed * 1000)).rand(dim).tolist()
    return EmbeddedChunk(
        vector=vec,
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=document_id,
        subject=subject,
        class_name=class_name,
        chapter=chapter,
        section="1.1 Cell Types",
        page_number=page,
        source_pdf=f"downloads/{subject}/{class_name}/book.pdf",
        chunk_number=1,
        total_chunks=1,
        token_count=8,
        character_count=len(text),
        original_text=text,
        metadata={"content_kind": "text"},
        embedding_model="test-model",
        embedding_timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def store() -> VectorStore:
    client = QdrantClient(":memory:")
    return VectorStore(config=QdrantConfig(batch_size=2), client=client)


def test_dynamic_collection_creation_no_hardcoded_subjects(store: VectorStore):
    assert store.list_collections() == []

    store.upsert_vectors([_make_chunk(subject="physics", document_id="doc-p1")])
    assert store.list_collections() == ["physics"]

    # A never-before-seen subject gets its own collection automatically.
    store.upsert_vectors([_make_chunk(subject="computer_science", document_id="doc-cs1")])
    assert set(store.list_collections()) == {"physics", "computer_science"}


def test_multi_subject_batch_partitions_into_separate_collections(store: VectorStore):
    chunks = [
        _make_chunk(subject="biology", document_id="doc-b1"),
        _make_chunk(subject="chemistry", document_id="doc-c1"),
        _make_chunk(subject="biology", document_id="doc-b1"),
    ]
    results = store.upsert_vectors(chunks)

    assert {r.collection_name for r in results} == {"biology", "chemistry"}
    assert store.count_vectors("biology") == 2
    assert store.count_vectors("chemistry") == 1


def test_reuploading_same_chunk_id_updates_not_duplicates(store: VectorStore):
    chunk_id = str(uuid.uuid4())
    store.upsert_vectors([_make_chunk(chunk_id=chunk_id, subject="biology", text="Original.", seed=0.1)])
    assert store.count_vectors("biology") == 1

    store.upsert_vectors([_make_chunk(chunk_id=chunk_id, subject="biology", text="Updated.", seed=0.9)])
    assert store.count_vectors("biology") == 1

    client = store.connect()
    point = client.retrieve(collection_name="biology", ids=[chunk_id], with_payload=True)[0]
    assert point.payload["text"] == "Updated."


def test_payload_contains_spec_keys_and_full_metadata(store: VectorStore):
    chunk = _make_chunk(subject="biology", document_id="doc-1")
    store.upsert_vectors([chunk])

    client = store.connect()
    payload = client.retrieve(collection_name="biology", ids=[chunk.chunk_id], with_payload=True)[0].payload

    expected_keys = {
        "chunk_id", "document_id", "subject", "class", "chapter", "section", "page",
        "source_pdf", "chunk_number", "total_chunks", "token_count", "embedding_model",
        "text", "metadata", "character_count", "embedding_timestamp",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["class"] == "class11"
    assert payload["text"] == chunk.original_text


def test_delete_document_scoped_to_that_document_only(store: VectorStore):
    chunks = [
        _make_chunk(subject="biology", document_id="doc-A"),
        _make_chunk(subject="biology", document_id="doc-A"),
        _make_chunk(subject="biology", document_id="doc-B"),
    ]
    store.upsert_vectors(chunks)
    assert store.count_vectors("biology") == 3

    deleted = store.delete_document("biology", "doc-A")
    assert deleted == 2
    assert store.count_vectors("biology") == 1


def test_delete_subject_removes_whole_collection(store: VectorStore):
    store.upsert_vectors([_make_chunk(subject="economics", document_id="doc-e1")])
    assert store.collection_exists("economics")

    assert store.delete_subject("economics") is True
    assert not store.collection_exists("economics")


def test_invalid_chunk_skipped_not_fatal(store: VectorStore):
    good = _make_chunk(subject="biology", document_id="doc-1")
    bad = _make_chunk(subject="biology", document_id="doc-1")
    bad.chunk_id = "not-a-valid-uuid"

    results = store.upsert_vectors([good, bad])

    assert results[0].requested == 2
    assert results[0].uploaded == 1
    assert results[0].skipped_invalid == 1


def test_batching_respects_configured_batch_size(store: VectorStore):
    chunks = [_make_chunk(subject="biology", document_id="doc-1") for _ in range(5)]
    results = store.upsert_vectors(chunks)

    assert results[0].batches == 3  # ceil(5/2) with batch_size=2


def test_vector_size_and_distance_inferred_from_actual_vectors(store: VectorStore):
    store.upsert_vectors([_make_chunk(subject="biology", dim=16)])

    info = store.get_collection("biology")
    assert info.config.params.vectors.size == 16
    assert info.config.params.vectors.distance == Distance.COSINE


def test_repeated_upserts_do_not_repeat_collection_existence_checks(store: VectorStore):
    # Regression guard for the CollectionManager existence-check cache: the
    # underlying client's collection_exists should only ever be hit once per
    # collection name, not once per upsert call.
    real_exists = store.connect().collection_exists
    calls = []

    def counting_exists(name):
        calls.append(name)
        return real_exists(name)

    store.connect().collection_exists = counting_exists

    for _ in range(5):
        store.upsert_vectors([_make_chunk(subject="biology", document_id="doc-1")])

    assert calls.count("biology") == 1
