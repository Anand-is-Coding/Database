"""Chunking module — responsible for splitting parsed documents into
retrieval-sized chunks using LlamaIndex node parsers.

Single responsibility: `Document` -> `list[Chunk]`. This module knows
nothing about embeddings, Qdrant, retrieval, or LLMs — it only converts one
already-parsed document at a time into semantically meaningful chunks.

Beyond LlamaIndex's default `SentenceSplitter`, this applies an
education-aware layer on top: tables, definitions, worked examples,
theorems, important notes, and formula+explanation pairs are treated as
atomic units that are never split, however small or large; everything else
is merged up to `target_chunk_size_tokens` and split by `SentenceSplitter`
at sentence/paragraph boundaries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from uuid import uuid4

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer

from chunking.exceptions import ChunkValidationError
from config.logging import get_logger
from models.chunk import Chunk, ChunkingConfig, ChunkMetadata, ContentKind
from models.document import (
    BulletList,
    ContentBlock,
    Document,
    Equation,
    ImageReference,
    OrderedList,
    Paragraph,
    Section,
    Table,
)

logger = get_logger(__name__)

# Patterns used to recognize textbook content that must be kept whole even
# though it's just regular `Paragraph` text as far as the parser is
# concerned. Matched against the start of the paragraph only, so body text
# that merely mentions "for example" mid-sentence isn't misclassified.
_DEFINITION_RE = re.compile(r"^\s*(definition|def\.)\s*[:\-–]?\s*\d*", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"^\s*(solved\s+)?example\s*[\-–:]?\s*\d*", re.IGNORECASE)
_THEOREM_RE = re.compile(r"^\s*(theorem|lemma|corollary)\s*[\-–:]?\s*\d*", re.IGNORECASE)
_NOTE_RE = re.compile(r"^\s*(important\s+)?note\s*[:\-–]", re.IGNORECASE)

_ATOMIC_KINDS = {ContentKind.DEFINITION, ContentKind.EXAMPLE, ContentKind.THEOREM, ContentKind.NOTE}


def _classify_paragraph(text: str) -> ContentKind:
    stripped = text.strip()
    if _DEFINITION_RE.match(stripped):
        return ContentKind.DEFINITION
    if _EXAMPLE_RE.match(stripped):
        return ContentKind.EXAMPLE
    if _THEOREM_RE.match(stripped):
        return ContentKind.THEOREM
    if _NOTE_RE.match(stripped):
        return ContentKind.NOTE
    return ContentKind.TEXT


@dataclass
class _Unit:
    """An intermediate, pre-chunk piece of content from one section's blocks."""

    text: str
    atomic: bool
    content_kind: ContentKind
    page_numbers: list[int] = field(default_factory=list)
    image_references: list[str] = field(default_factory=list)


@dataclass
class _RawPiece:
    """A finalized chunk of text, still missing sequence-wide metadata
    (`chunk_number`/`total_chunks`) that can only be assigned once every
    section in the document has been processed.
    """

    text: str
    content_kind: ContentKind
    page_numbers: list[int]
    image_references: list[str]
    chapter: str | None
    section: str | None


@dataclass
class ChunkValidationReport:
    """Summary produced by `LlamaChunker.validate_chunks`."""

    total_chunks: int
    empty_chunks: int
    duplicate_chunks: int
    oversized_chunks: int
    undersized_chunks: int
    missing_metadata: int

    @property
    def is_valid(self) -> bool:
        return self.empty_chunks == 0 and self.missing_metadata == 0


class LlamaChunker:
    """Splits a parsed `Document` into semantically meaningful `Chunk`s.

    Stateless per call (holds only its config/splitter, set once at
    construction) — a single instance can chunk multiple documents
    sequentially, or multiple instances can be used across threads/processes
    for parallel batch chunking without any shared mutable state.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: object | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._tokenizer = tokenizer or get_tokenizer()
        self._splitter = SentenceSplitter(
            chunk_size=self.config.target_chunk_size_tokens,
            chunk_overlap=self.config.chunk_overlap_tokens,
            tokenizer=self._tokenizer,
            paragraph_separator="\n\n",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, document: Document) -> list[Chunk]:
        """Convert a parsed `Document` into an ordered list of `Chunk`s."""
        logger.info(
            "Document received: %s (%d top-level section(s))",
            document.metadata.source_path,
            len(document.sections),
        )

        raw_pieces: list[_RawPiece] = []
        for top_section in document.sections:
            self._walk_section(top_section, chapter=top_section.heading, raw_pieces=raw_pieces)

        chunks = self._finalize_chunks(raw_pieces, document)
        self._log_summary(document, chunks)
        self.validate_chunks(chunks)
        return chunks

    def validate_chunks(self, chunks: list[Chunk]) -> ChunkValidationReport:
        """Validate generated chunks and log a summary.

        Raises `ChunkValidationError` for structural defects (empty text,
        missing required metadata). Size/duplicate issues are logged as
        warnings rather than treated as hard failures, since a handful of
        small leftover chunks or a legitimately repeated table row isn't
        necessarily a bug.
        """
        seen_hashes: set[str] = set()
        empty = duplicate = oversized = undersized = missing_metadata = 0
        max_allowed_tokens = self.config.target_chunk_size_tokens * 2

        for c in chunks:
            if not c.text or not c.text.strip():
                empty += 1
                continue

            digest = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                duplicate += 1
            seen_hashes.add(digest)

            if not c.metadata.document_id or not c.metadata.source_pdf:
                missing_metadata += 1

            if (
                c.metadata.estimated_token_count > max_allowed_tokens
                and c.metadata.content_kind not in _ATOMIC_KINDS
                and c.metadata.content_kind != ContentKind.TABLE
            ):
                oversized += 1
            if c.metadata.estimated_token_count < self.config.min_chunk_size_tokens:
                undersized += 1

        report = ChunkValidationReport(
            total_chunks=len(chunks),
            empty_chunks=empty,
            duplicate_chunks=duplicate,
            oversized_chunks=oversized,
            undersized_chunks=undersized,
            missing_metadata=missing_metadata,
        )

        logger.info(
            "Validation results: %d total, %d empty, %d duplicate, %d oversized, "
            "%d undersized, %d missing metadata",
            report.total_chunks,
            report.empty_chunks,
            report.duplicate_chunks,
            report.oversized_chunks,
            report.undersized_chunks,
            report.missing_metadata,
        )
        if duplicate:
            logger.warning("%d duplicate chunk(s) detected.", duplicate)
        if oversized:
            logger.warning("%d chunk(s) exceed the configured size limit.", oversized)
        if undersized:
            logger.warning("%d chunk(s) are smaller than min_chunk_size_tokens.", undersized)

        if not report.is_valid:
            raise ChunkValidationError(
                f"Chunk validation failed: {empty} empty chunk(s), "
                f"{missing_metadata} chunk(s) missing required metadata."
            )
        return report

    # ------------------------------------------------------------------
    # Section traversal
    # ------------------------------------------------------------------

    def _walk_section(
        self, section: Section, chapter: str | None, raw_pieces: list[_RawPiece]
    ) -> None:
        section_heading = section.heading or chapter
        units = self._build_units(section.blocks)
        pieces = self._units_to_pieces(units)

        for text, kind, pages, image_refs in pieces:
            raw_pieces.append(
                _RawPiece(
                    text=text,
                    content_kind=kind,
                    page_numbers=pages,
                    image_references=image_refs,
                    chapter=chapter,
                    section=section_heading,
                )
            )
        if pieces:
            logger.info(
                "Section processed: chapter='%s' section='%s' -> %d chunk piece(s)",
                chapter or "(untitled)",
                section_heading or "(untitled)",
                len(pieces),
            )

        # Chapter is pinned to the top-level section's own heading for the
        # whole subtree; only `section_heading` changes as we go deeper.
        for sub in section.subsections:
            self._walk_section(sub, chapter=chapter, raw_pieces=raw_pieces)

    # ------------------------------------------------------------------
    # Block -> unit grouping (atomic vs. splittable)
    # ------------------------------------------------------------------

    def _build_units(self, blocks: list[ContentBlock]) -> list[_Unit]:
        units: list[_Unit] = []
        i = 0
        n = len(blocks)
        while i < n:
            block = blocks[i]

            if isinstance(block, Table):
                units.append(self._table_unit(block))
                i += 1
                continue

            if isinstance(block, ImageReference):
                units.append(self._image_unit(block))
                i += 1
                continue

            if isinstance(block, Equation):
                consumed = self._append_equation_unit(block, blocks, i, units)
                i += consumed
                continue

            if isinstance(block, (OrderedList, BulletList)):
                units.append(self._list_unit(block))
                i += 1
                continue

            if isinstance(block, Paragraph):
                kind = _classify_paragraph(block.text)
                units.append(
                    _Unit(
                        text=block.text,
                        atomic=kind in _ATOMIC_KINDS,
                        content_kind=kind,
                        page_numbers=_pages(block.page_number),
                    )
                )
                i += 1
                continue

            i += 1  # defensive: parser only emits the block types handled above

        return units

    @staticmethod
    def _append_equation_unit(
        equation: Equation, blocks: list[ContentBlock], index: int, units: list[_Unit]
    ) -> int:
        """Bundle a formula with its explanation so they're never separated.

        Prefers the paragraph immediately after the equation; falls back to
        merging into the immediately preceding splittable unit; otherwise
        keeps the formula as its own atomic unit.
        """
        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        if isinstance(next_block, Paragraph):
            units.append(
                _Unit(
                    text=f"{equation.text}\n{next_block.text}",
                    atomic=True,
                    content_kind=ContentKind.FORMULA,
                    page_numbers=_pages(equation.page_number, next_block.page_number),
                )
            )
            return 2

        if units and not units[-1].atomic:
            prev = units[-1]
            prev.text = f"{prev.text}\n{equation.text}"
            prev.atomic = True
            prev.content_kind = ContentKind.FORMULA
            prev.page_numbers = sorted(set(prev.page_numbers) | set(_pages(equation.page_number)))
            return 1

        units.append(
            _Unit(
                text=equation.text,
                atomic=True,
                content_kind=ContentKind.FORMULA,
                page_numbers=_pages(equation.page_number),
            )
        )
        return 1

    @staticmethod
    def _table_unit(table: Table) -> _Unit:
        lines = [table.caption] if table.caption else []
        lines.extend(" | ".join(cell.text for cell in row) for row in table.rows)
        return _Unit(
            text="\n".join(lines),
            atomic=True,
            content_kind=ContentKind.TABLE,
            page_numbers=_pages(table.page_number),
        )

    @staticmethod
    def _image_unit(image: ImageReference) -> _Unit:
        placeholder = f"[IMAGE: {image.caption or image.reference_id}]"
        return _Unit(
            text=placeholder,
            atomic=True,
            content_kind=ContentKind.IMAGE,
            page_numbers=_pages(image.page_number),
            image_references=[image.reference_id],
        )

    @staticmethod
    def _list_unit(block: OrderedList | BulletList) -> _Unit:
        if isinstance(block, OrderedList):
            text = "\n".join(f"{idx}. {item}" for idx, item in enumerate(block.items, start=1))
        else:
            text = "\n".join(f"- {item}" for item in block.items)
        return _Unit(
            text=text, atomic=False, content_kind=ContentKind.TEXT, page_numbers=_pages(block.page_number)
        )

    # ------------------------------------------------------------------
    # Unit -> chunk-sized pieces
    # ------------------------------------------------------------------

    def _units_to_pieces(
        self, units: list[_Unit]
    ) -> list[tuple[str, ContentKind, list[int], list[str]]]:
        """Merge consecutive splittable units and hand them to `SentenceSplitter`;
        atomic units pass through untouched as their own single piece.

        A split piece is attributed the full page range of the merged run it
        came from (paragraph-level page granularity, not per-sentence), so
        page numbers on split pieces are a reasonable approximation rather
        than an exact per-character mapping.
        """
        pieces: list[tuple[str, ContentKind, list[int], list[str]]] = []
        buffer_texts: list[str] = []
        buffer_pages: set[int] = set()

        def flush_buffer() -> None:
            if not buffer_texts:
                return
            combined = "\n\n".join(buffer_texts)
            pages = sorted(buffer_pages)
            for split_text in self._splitter.split_text(combined):
                if split_text.strip():
                    pieces.append((split_text, ContentKind.TEXT, pages, []))
            buffer_texts.clear()
            buffer_pages.clear()

        for unit in units:
            if unit.atomic:
                flush_buffer()
                pieces.append((unit.text, unit.content_kind, unit.page_numbers, unit.image_references))
            else:
                buffer_texts.append(unit.text)
                buffer_pages.update(unit.page_numbers)

        flush_buffer()
        return pieces

    # ------------------------------------------------------------------
    # Final chunk assembly
    # ------------------------------------------------------------------

    def _finalize_chunks(self, raw_pieces: list[_RawPiece], document: Document) -> list[Chunk]:
        total = len(raw_pieces)
        chunks: list[Chunk] = []

        for idx, piece in enumerate(raw_pieces, start=1):
            text = self._apply_heading_context(piece, document)
            chunk_id = str(uuid4())
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document.metadata.document_id,
                subject=document.metadata.subject,
                class_name=document.metadata.class_name,
                chapter=piece.chapter,
                section=piece.section,
                page_number=piece.page_numbers[0] if piece.page_numbers else None,
                page_numbers=piece.page_numbers,
                source_pdf=document.metadata.source_path,
                parser_version=document.metadata.parser_version,
                chunk_number=idx,
                total_chunks=total,
                character_count=len(text),
                estimated_token_count=len(self._tokenizer(text)),
                content_kind=piece.content_kind,
                image_references=piece.image_references,
            )
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))

        return chunks

    def _apply_heading_context(self, piece: _RawPiece, document: Document) -> str:
        if not self.config.include_heading_context:
            return piece.text
        section_label = piece.section if piece.section != piece.chapter else None
        breadcrumb_parts = [p for p in (document.metadata.subject, piece.chapter, section_label) if p]
        if not breadcrumb_parts:
            return piece.text
        return f"[{' > '.join(breadcrumb_parts)}]\n{piece.text}"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(document: Document, chunks: list[Chunk]) -> None:
        if chunks:
            avg_chars = sum(c.metadata.character_count for c in chunks) / len(chunks)
            avg_tokens = sum(c.metadata.estimated_token_count for c in chunks) / len(chunks)
        else:
            avg_chars = avg_tokens = 0.0
        logger.info(
            "Chunks generated: %d (avg %.0f chars / %.0f tokens) for %s",
            len(chunks),
            avg_chars,
            avg_tokens,
            document.metadata.source_path,
        )


def _pages(*page_numbers: int | None) -> list[int]:
    return sorted({p for p in page_numbers if p is not None})
