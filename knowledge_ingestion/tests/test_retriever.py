"""Unit tests for Retriever.

Filtering/collection-selection/ranking logic is tested with a fake,
deterministic embedding model (fast, no network) driving a real in-memory
Qdrant. A separate @integration test uses a real small model to verify
genuine semantic relevance end to end.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from qdrant_client import QdrantClient

from embedding.bge_embedder import EmbeddingService
from models.embedded_chunk import EmbeddedChunk, EmbeddingConfig
from models.qdrant import QdrantConfig
from models.retrieval import RetrievalConfig, TeachingContext
from retrieval.exceptions import CollectionNotFoundError, QueryEmbeddingError
from retrieval.retriever import Retriever
from vectorstore.qdrant_client import VectorStore


class _DeterministicFakeModel:
    """Maps each distinct text to a fixed pseudo-random unit vector, so the
    same text always embeds to the same vector (deterministic tests) without
    downloading a real model.
    """

    DIM = 16

    def get_embedding_dimension(self) -> int:
        return self.DIM

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            vec = rng.rand(self.DIM).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors)


def _embed(text: str) -> list[float]:
    return _DeterministicFakeModel().encode([text])[0].tolist()


def _make_chunk(text, subject, document_id="doc-1", chapter="Chapter 1: The Cell", class_name="class11",
                 section="1.1 Cell Types", page=1):
    return EmbeddedChunk(
        vector=_embed(text),
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        subject=subject,
        class_name=class_name,
        chapter=chapter,
        section=section,
        page_number=page,
        source_pdf=f"downloads/{subject}/{class_name}/book.pdf",
        chunk_number=1,
        total_chunks=1,
        token_count=8,
        character_count=len(text),
        original_text=text,
        metadata={"content_kind": "text"},
        embedding_model="fake-model",
        embedding_timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def embedder() -> EmbeddingService:
    return EmbeddingService(config=EmbeddingConfig(), model=_DeterministicFakeModel())


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(config=QdrantConfig(), client=QdrantClient(":memory:"))


@pytest.fixture
def populated_store(store: VectorStore) -> VectorStore:
    biology_chunks = [
        _make_chunk("The cell is the basic unit of life.", "biology", chapter="Chapter 1: The Cell", page=1),
        _make_chunk("Photosynthesis converts sunlight into energy.", "biology", chapter="Chapter 2: Plants", page=20),
        _make_chunk("Mitochondria produce ATP.", "biology", chapter="Chapter 1: The Cell", page=5, class_name="class12"),
    ]
    physics_chunks = [
        _make_chunk("Newton's second law relates force and mass.", "physics", chapter="Chapter 3: Motion", page=40),
    ]
    store.upsert_vectors(biology_chunks + physics_chunks)
    return store


def test_known_subject_scopes_search_to_that_collection_only(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=5))
    context = TeachingContext(subject="biology", student_question="cells")

    results = retriever.retrieve(context)

    assert len(results) == 3
    assert all(r.subject == "biology" for r in results)


def test_unknown_subject_searches_all_collections(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=10))
    context = TeachingContext(student_question="motion")

    results = retriever.retrieve(context)

    subjects_seen = {r.subject for r in results}
    assert "physics" in subjects_seen and "biology" in subjects_seen


def test_chapter_filter_narrows_results(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=5))
    context = TeachingContext(subject="biology", chapter="Chapter 1: The Cell", student_question="cells")

    results = retriever.retrieve(context)

    assert len(results) == 2
    assert all(r.chapter == "Chapter 1: The Cell" for r in results)


def test_class_filter_narrows_results(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=5))
    context = TeachingContext(subject="biology", class_name="class12", student_question="anything")

    results = retriever.retrieve(context)

    assert len(results) == 1
    assert results[0].class_name == "class12"


def test_page_range_filter_via_config(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(
        vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=5, page_min=15, page_max=25)
    )
    context = TeachingContext(subject="biology", student_question="anything")

    results = retriever.retrieve(context)

    assert len(results) == 1
    assert 15 <= results[0].page_number <= 25


def test_missing_collection_returns_empty_via_retrieve(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder)
    context = TeachingContext(subject="history", student_question="anything")

    assert retriever.retrieve(context) == []


def test_missing_collection_raises_via_retrieve_by_subject(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder)
    context = TeachingContext(subject="history", student_question="anything")

    with pytest.raises(CollectionNotFoundError):
        retriever.retrieve_by_subject(context)


def test_empty_question_rejected(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder)
    context = TeachingContext(subject="biology", student_question="   ")

    with pytest.raises(QueryEmbeddingError):
        retriever.retrieve(context)


def test_topic_blending_changes_query_text(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder)
    plain = TeachingContext(subject="biology", student_question="why does it happen?")
    with_topic = TeachingContext(subject="biology", current_topic="Photosynthesis", student_question="why does it happen?")

    assert retriever._build_query_text(plain) == "why does it happen?"
    assert retriever._build_query_text(with_topic) == "Topic: Photosynthesis\nQuestion: why does it happen?"


def test_no_duplicate_chunk_ids_in_merged_results(populated_store: VectorStore, embedder: EmbeddingService):
    retriever = Retriever(vector_store=populated_store, embedding_service=embedder, config=RetrievalConfig(top_k=10))
    context = TeachingContext(student_question="anything")

    results = retriever.retrieve(context)
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids))


@pytest.mark.integration
def test_real_model_semantic_relevance_powerhouse_of_the_cell():
    client = QdrantClient(":memory:")
    store = VectorStore(config=QdrantConfig(), client=client)
    embedder = EmbeddingService(config=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    def embed(text: str) -> list[float]:
        return embedder.model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].tolist()

    chunks = [
        EmbeddedChunk(
            vector=embed("The cell is the basic unit of life."),
            chunk_id=str(uuid.uuid4()), document_id="doc-1", subject="biology", class_name="class11",
            chapter="Chapter 1", section="1.1", page_number=1, source_pdf="x.pdf", chunk_number=1,
            total_chunks=2, token_count=8, character_count=10, original_text="The cell is the basic unit of life.",
            embedding_model="test", embedding_timestamp=datetime.now(timezone.utc),
        ),
        EmbeddedChunk(
            vector=embed("Mitochondria are the powerhouse of the cell, producing ATP."),
            chunk_id=str(uuid.uuid4()), document_id="doc-1", subject="biology", class_name="class11",
            chapter="Chapter 1", section="1.2", page_number=5, source_pdf="x.pdf", chunk_number=2,
            total_chunks=2, token_count=8, character_count=10, original_text="Mitochondria are the powerhouse of the cell, producing ATP.",
            embedding_model="test", embedding_timestamp=datetime.now(timezone.utc),
        ),
    ]
    store.upsert_vectors(chunks)

    retriever = Retriever(vector_store=store, embedding_service=embedder, config=RetrievalConfig(top_k=1))
    context = TeachingContext(subject="biology", student_question="What is the powerhouse of the cell?")
    results = retriever.retrieve(context)

    assert len(results) == 1
    assert "Mitochondria" in results[0].text
