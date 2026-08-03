"""Entrypoint for the Knowledge Ingestion Pipeline.

Constructs the pipeline's dependencies and triggers a run. No business logic
lives here — this module only wires the composition root together.
"""

from __future__ import annotations

import os

# Strictly constrain all underlying C++ threading backends to 1 thread
# to prevent std::bad_alloc on memory-constrained machines.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from chunking.llama_chunker import LlamaChunker
from config.logging import get_logger
from embedding.bge_embedder import EmbeddingService
from loaders.s3_loader import S3Loader
from parser.docling_parser import DoclingParser
from pipeline.ingest_pipeline import IngestionPipeline
from vectorstore.qdrant_client import VectorStore

logger = get_logger(__name__)


def build_pipeline() -> IngestionPipeline:
    """Composition root: instantiate and wire all pipeline dependencies."""
    return IngestionPipeline(
        loader=S3Loader(),
        parser=DoclingParser(),
        chunker=LlamaChunker(),
        embedder=EmbeddingService(),
        vector_store=VectorStore(),
    )


def main() -> None:
    logger.info("Starting Knowledge Ingestion Pipeline...")
    pipeline = build_pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
