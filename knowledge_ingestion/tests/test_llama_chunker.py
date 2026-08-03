"""Unit tests for LlamaChunker — real pydantic models and the real
llama-index-core SentenceSplitter (lightweight, no ML model download).
"""

from __future__ import annotations

import pytest

from chunking.llama_chunker import LlamaChunker
from models.chunk import ContentKind
from models.document import (
    BulletList,
    Document,
    Equation,
    ImageReference,
    Metadata,
    OrderedList,
    Page,
    Paragraph,
    Section,
    Table,
    TableCell,
)


def _long_paragraph(n_sentences: int, seed: str) -> str:
    return " ".join(
        f"{seed} sentence number {i} discusses cellular respiration and energy production in mitochondria."
        for i in range(n_sentences)
    )


@pytest.fixture
def sample_document() -> Document:
    meta = Metadata(
        subject="biology",
        class_name="class11",
        book_name="chapter1",
        source_path="downloads/biology/class11/chapter1.pdf",
        total_pages=3,
        parser_version="test",
    )

    blocks = [
        Paragraph(text=_long_paragraph(15, "Intro"), page_number=1, reading_order=0),
        Paragraph(
            text="Definition: A cell is the smallest structural and functional unit of an organism.",
            page_number=1,
            reading_order=1,
        ),
        Paragraph(
            text="Example 1: Consider a red blood cell. " * 20,
            page_number=1,
            reading_order=2,
        ),
        Paragraph(
            text="Theorem: Cell theory states that all living organisms are composed of cells.",
            page_number=1,
            reading_order=3,
        ),
        Paragraph(
            text="Important Note: Viruses are not considered cells.",
            page_number=1,
            reading_order=4,
        ),
        Equation(text="ATP -> ADP + Pi + energy", page_number=2, reading_order=5),
        Paragraph(
            text="This reaction releases the energy stored in the phosphate bond of ATP.",
            page_number=2,
            reading_order=6,
        ),
        OrderedList(items=["Prokaryotic cells", "Eukaryotic cells"], page_number=2, reading_order=7),
        BulletList(items=["No nucleus", "Simpler structure"], page_number=2, reading_order=8),
        Table(
            caption="Table 1: Cell comparison",
            rows=[
                [TableCell(text="Feature", is_header=True), TableCell(text="Prokaryote", is_header=True)],
                [TableCell(text="Nucleus"), TableCell(text="Absent")],
            ],
            num_rows=2,
            num_cols=2,
            page_number=2,
            reading_order=9,
        ),
        ImageReference(reference_id="#/pictures/0", caption="Figure 1: A cell diagram", page_number=2, reading_order=10),
        Paragraph(text=_long_paragraph(20, "Trailing"), page_number=3, reading_order=11),
    ]

    subsection = Section(heading="1.1 Cell Types", level=2, page_number=1, blocks=blocks)
    chapter = Section(heading="Chapter 1: The Cell", level=1, page_number=1, subsections=[subsection])

    return Document(
        metadata=meta,
        title="Biology Textbook",
        pages=[Page(page_number=i, width=595.0, height=842.0) for i in (1, 2, 3)],
        sections=[chapter],
    )


@pytest.fixture
def chunker() -> LlamaChunker:
    from models.chunk import ChunkingConfig

    return LlamaChunker(config=ChunkingConfig(target_chunk_size_tokens=120, chunk_overlap_tokens=20))


def test_definition_example_theorem_note_kept_whole(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)

    by_kind = {kind: [c for c in chunks if c.metadata.content_kind == kind] for kind in
               (ContentKind.DEFINITION, ContentKind.EXAMPLE, ContentKind.THEOREM, ContentKind.NOTE)}

    assert len(by_kind[ContentKind.DEFINITION]) == 1
    assert len(by_kind[ContentKind.EXAMPLE]) == 1
    assert len(by_kind[ContentKind.THEOREM]) == 1
    assert len(by_kind[ContentKind.NOTE]) == 1

    # The repeated "Example" paragraph is long enough to exceed the 120-token
    # target — it must still be kept as one chunk, never split.
    assert by_kind[ContentKind.EXAMPLE][0].metadata.estimated_token_count > 120


def test_equation_merged_with_explanation(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    formula_chunks = [c for c in chunks if c.metadata.content_kind == ContentKind.FORMULA]

    assert len(formula_chunks) == 1
    assert "ATP -> ADP" in formula_chunks[0].text
    assert "releases the energy" in formula_chunks[0].text


def test_table_kept_whole_not_flattened(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    table_chunks = [c for c in chunks if c.metadata.content_kind == ContentKind.TABLE]

    assert len(table_chunks) == 1
    assert "Feature | Prokaryote" in table_chunks[0].text
    assert "Nucleus | Absent" in table_chunks[0].text


def test_image_replaced_with_placeholder_reference_preserved(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    image_chunks = [c for c in chunks if c.metadata.content_kind == ContentKind.IMAGE]

    assert len(image_chunks) == 1
    assert "[IMAGE:" in image_chunks[0].text
    assert image_chunks[0].metadata.image_references == ["#/pictures/0"]


def test_long_paragraphs_split_by_sentence_splitter(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    text_chunks = [c for c in chunks if c.metadata.content_kind == ContentKind.TEXT]

    assert len(text_chunks) > 2
    for c in text_chunks:
        assert c.metadata.estimated_token_count <= 120 * 1.5


def test_heading_context_breadcrumb_prefixed(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    assert chunks[0].text.startswith("[biology > Chapter 1: The Cell")


def test_chunk_number_and_total_chunks_consistent(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)

    assert [c.metadata.chunk_number for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.metadata.total_chunks == len(chunks) for c in chunks)


def test_validate_chunks_reports_valid_with_no_empty_or_missing_metadata(chunker: LlamaChunker, sample_document: Document):
    chunks = chunker.chunk(sample_document)
    report = chunker.validate_chunks(chunks)

    assert report.is_valid
    assert report.empty_chunks == 0
    assert report.missing_metadata == 0
