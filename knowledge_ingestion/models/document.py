"""Structured document model produced by the parser stage.

Represents a parsed PDF as a hierarchy of `Section`s containing ordered
content blocks (paragraphs, lists, tables, images, equations) — preserving
reading order and section nesting instead of flattening to plain text.
This is the contract the chunking module (`chunking/llama_chunker.py`) consumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from config.settings import settings


class ParserConfig(BaseModel):
    """Tunable knobs for `DoclingParser`.

    `do_ocr`/`do_table_structure` default from `Settings`, so disabling OCR
    for a memory-constrained machine (OCR is one of the heaviest parts of
    Docling's pipeline) is a `.env` change, not a code change. Defaults
    match Docling's own defaults, so an unconfigured `DoclingParser()`
    behaves exactly as before this config existed.
    """

    do_ocr: bool = Field(default_factory=lambda: settings.DOCLING_DO_OCR)
    do_table_structure: bool = Field(default_factory=lambda: settings.DOCLING_DO_TABLE_STRUCTURE)
    num_threads: int = Field(default_factory=lambda: settings.DOCLING_NUM_THREADS)
    batch_size: int = Field(default_factory=lambda: settings.DOCLING_BATCH_SIZE)
    image_scale: float = Field(default_factory=lambda: settings.DOCLING_IMAGE_SCALE)


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    ORDERED_LIST = "ordered_list"
    BULLET_LIST = "bullet_list"
    TABLE = "table"
    IMAGE = "image"
    EQUATION = "equation"


class Paragraph(BaseModel):
    """A block of body text — also used for captions not linked to a table/image."""

    type: Literal[BlockType.PARAGRAPH] = BlockType.PARAGRAPH
    text: str
    page_number: int | None = None
    reading_order: int


class OrderedList(BaseModel):
    """A numbered/lettered list (e.g. `1. 2. 3.`)."""

    type: Literal[BlockType.ORDERED_LIST] = BlockType.ORDERED_LIST
    items: list[str]
    page_number: int | None = None
    reading_order: int


class BulletList(BaseModel):
    """An unordered (bulleted) list."""

    type: Literal[BlockType.BULLET_LIST] = BlockType.BULLET_LIST
    items: list[str]
    page_number: int | None = None
    reading_order: int


class Equation(BaseModel):
    """A mathematical formula, captured as text/LaTeX when Docling recognizes it."""

    type: Literal[BlockType.EQUATION] = BlockType.EQUATION
    text: str
    page_number: int | None = None
    reading_order: int


class TableCell(BaseModel):
    """A single cell in a `Table`'s reconstructed grid, with its row/col span."""

    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


class Table(BaseModel):
    """A table, preserved as a row-major grid of cells — never flattened to text."""

    type: Literal[BlockType.TABLE] = BlockType.TABLE
    caption: str | None = None
    rows: list[list[TableCell]]
    num_rows: int
    num_cols: int
    page_number: int | None = None
    reading_order: int


class ImageReference(BaseModel):
    """A pointer to an image/chart on the page — never the image bytes themselves."""

    type: Literal[BlockType.IMAGE] = BlockType.IMAGE
    reference_id: str
    caption: str | None = None
    page_number: int | None = None
    reading_order: int
    width: float | None = None
    height: float | None = None


ContentBlock = Annotated[
    Union[Paragraph, OrderedList, BulletList, Table, ImageReference, Equation],
    Field(discriminator="type"),
]


class Section(BaseModel):
    """A heading and everything under it, up to the next heading at the same or a
    shallower level. `blocks` preserves reading order within the section;
    `subsections` preserves nesting for headings below it.
    """

    heading: str | None = None
    level: int = 0
    page_number: int | None = None
    blocks: list[ContentBlock] = Field(default_factory=list)
    subsections: list["Section"] = Field(default_factory=list)


Section.model_rebuild()


class Page(BaseModel):
    """One physical page of the source PDF."""

    page_number: int
    width: float | None = None
    height: float | None = None


class Metadata(BaseModel):
    """Traces a parsed document back to its S3/local origin and parse context."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str | None = None
    class_name: str | None = None
    book_name: str
    source_path: str
    total_pages: int
    parser_name: str = "docling"
    parser_version: str
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Document(BaseModel):
    """A fully parsed PDF, ready to be handed to the chunking module."""

    metadata: Metadata
    title: str | None = None
    pages: list[Page] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
