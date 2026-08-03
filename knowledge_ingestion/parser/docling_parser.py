"""PDF parsing module — responsible for converting raw PDFs into structured
document representations using Docling.

Single responsibility: PDF -> structured `Document`. This module knows
nothing about chunking, embeddings, Qdrant, retrieval, or LLMs — it only
converts one file at a time and hands back a strongly typed document tree.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.picture.picture import PictureItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import FormulaItem, ListItem, SectionHeaderItem, TitleItem
from docling_core.types.doc.labels import DocItemLabel

from config.logging import get_logger
from models.document import (
    BulletList,
    Document,
    Equation,
    ImageReference,
    Metadata,
    OrderedList,
    Page,
    Paragraph,
    ParserConfig,
    Section,
    Table,
    TableCell,
)
from parser.exceptions import (
    CorruptedPDFError,
    DocumentValidationError,
    EmptyDocumentError,
    ParserError,
    PasswordProtectedPDFError,
    UnsupportedDocumentError,
)

logger = get_logger(__name__)

# Labels expected to become Paragraph blocks. Anything else unrecognized
# still falls back to Paragraph in `extract_structure` (with a warning)
# rather than being silently dropped.
_TEXT_BLOCK_LABELS = {
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TEXT,
    DocItemLabel.FOOTNOTE,
    DocItemLabel.REFERENCE,
    DocItemLabel.CAPTION,
    DocItemLabel.HANDWRITTEN_TEXT,
    DocItemLabel.CODE,
}


def _docling_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:
        try:
            return version("docling-core")
        except PackageNotFoundError:
            return "unknown"


class DoclingParser:
    """Parses PDF files into structured, layout-aware `Document` objects.

    The Docling converter is injected (or lazily built on first use) so
    tests can supply a stub/mocked converter instead of exercising the real
    ML-backed conversion pipeline.
    """

    def __init__(self, config: ParserConfig | None = None, converter: Any | None = None) -> None:
        self.config = config or ParserConfig()
        self._converter = converter
        self._parser_version = _docling_version()

    @property
    def converter(self) -> Any:
        if self._converter is None:
            self._converter = self._build_default_converter(self.config)
        return self._converter

    @staticmethod
    def _build_default_converter(config: ParserConfig) -> Any:
        """Lazily import and build the real Docling conversion pipeline.

        Imported here (not at module level) so this module — and its
        extraction logic — stays importable/testable even in environments
        that only have `docling-core` installed, not the full `docling`
        package with its ML backends.

        `config.do_ocr`/`do_table_structure` are passed straight through to
        Docling's own `PdfPipelineOptions` - OCR in particular is one of the
        heaviest parts of Docling's pipeline (memory and time), and is
        unnecessary for digitally-generated PDFs with a real text layer
        (most textbooks); disabling it is the single biggest lever for
        running this on a memory-constrained machine.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = config.do_ocr
        pipeline_options.do_table_structure = config.do_table_structure
        
        # Memory constraint settings
        pipeline_options.accelerator_options.num_threads = config.num_threads
        pipeline_options.layout_batch_size = config.batch_size
        pipeline_options.ocr_batch_size = config.batch_size
        pipeline_options.table_batch_size = config.batch_size
        pipeline_options.images_scale = config.image_scale

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        pdf_path: Path | str,
        *,
        subject: str | None = None,
        class_name: str | None = None,
    ) -> Document:
        """Parse a single PDF file and return its structured representation.

        `subject`/`class_name` are optional explicit overrides. Pass them
        when the caller already knows the true values (e.g. `IngestionPipeline`
        knows them precisely from the `S3PdfObject` the loader discovered) -
        this is strictly more reliable than re-deriving them from the local
        download path, which only works when that path happens to be exactly
        two levels deep (`<subject>/<class>/<book>.pdf`). A flatter S3 layout
        (`<subject>/<book>.pdf`, no class subfolder) is indistinguishable
        from a deeper one by path shape alone, so leave both `None` only for
        standalone/direct use (e.g. `parse_directory`), where path inference
        is the best information available.
        """
        pdf_path = Path(pdf_path)
        logger.info("Parsing started: %s", pdf_path)

        if not pdf_path.exists():
            raise UnsupportedDocumentError(f"File does not exist: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedDocumentError(f"Not a PDF file: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise EmptyDocumentError(f"File is empty (0 bytes): {pdf_path}")

        docling_doc = self._convert(pdf_path)

        metadata = self.extract_document_metadata(
            pdf_path, docling_doc, subject=subject, class_name=class_name
        )
        title, sections, warnings = self.extract_structure(docling_doc)
        pages = self._extract_pages(docling_doc)

        document = Document(
            metadata=metadata,
            title=title,
            pages=pages,
            sections=sections,
            warnings=warnings,
        )
        self.validate_document(document)

        logger.info(
            "Parsing completed: %s (%d page(s), %d top-level section(s), %d warning(s))",
            pdf_path,
            len(pages),
            len(sections),
            len(warnings),
        )
        return document

    def parse_directory(self, directory: Path | str) -> Iterator[Document]:
        """Parse every PDF found (recursively) under `directory`, one at a time.

        A single corrupted/unsupported file logs an error and is skipped so
        it doesn't abort the rest of the batch. This is a thin convenience
        wrapper — a real batch pipeline (parallelism, retries, progress
        tracking) belongs in a future orchestration stage, not here.
        """
        directory = Path(directory)
        pdf_paths = sorted(directory.rglob("*.pdf"))
        logger.info("Discovered %d PDF file(s) under %s", len(pdf_paths), directory)

        for pdf_path in pdf_paths:
            try:
                yield self.parse(pdf_path)
            except ParserError as exc:
                logger.error("Skipping '%s' due to parsing failure: %s", pdf_path, exc)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _convert(self, pdf_path: Path) -> DoclingDocument:
        try:
            result = self.converter.convert(pdf_path)
        except Exception as exc:  # noqa: BLE001 - backend raises various types
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise PasswordProtectedPDFError(f"PDF is password-protected: {pdf_path}") from exc
            raise CorruptedPDFError(f"Failed to parse PDF '{pdf_path}': {exc}") from exc

        status_name = getattr(result.status, "name", str(result.status)).upper()
        if status_name == "FAILURE":
            error_messages = "; ".join(getattr(e, "error_message", str(e)) for e in result.errors)
            if "password" in error_messages.lower() or "encrypt" in error_messages.lower():
                raise PasswordProtectedPDFError(
                    f"PDF is password-protected: {pdf_path} ({error_messages})"
                )
            raise CorruptedPDFError(f"Docling failed to convert '{pdf_path}': {error_messages}")

        if status_name == "PARTIAL_SUCCESS":
            logger.warning("Partial conversion for '%s' — some content may be missing.", pdf_path)

        return result.document

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def extract_document_metadata(
        self,
        pdf_path: Path,
        doc: DoclingDocument,
        *,
        subject: str | None = None,
        class_name: str | None = None,
    ) -> Metadata:
        """Build `Metadata`. Uses `subject`/`class_name` directly when given;
        otherwise falls back to inferring them from the folder hierarchy.

        Path-based inference expects `<subject>/<class>/<book>.pdf` — the
        immediate parent directory is treated as the class, and its parent
        as the subject. Deliberately relative (not anchored to a literal
        "downloads/" prefix) so this stays decoupled from wherever the
        loader stage put the files — but it's still a guess, and a wrong one
        whenever the real layout isn't exactly two levels deep. Prefer
        passing `subject`/`class_name` explicitly whenever the caller
        already knows them (see `parse()`).
        """
        if subject is None and class_name is None:
            subject, class_name = self._infer_subject_class(pdf_path)
        total_pages = self._safe_num_pages(doc)

        return Metadata(
            subject=subject,
            class_name=class_name,
            book_name=pdf_path.stem,
            source_path=str(pdf_path),
            total_pages=total_pages,
            parser_version=self._parser_version,
        )

    @staticmethod
    def _infer_subject_class(pdf_path: Path) -> tuple[str | None, str | None]:
        class_dir = pdf_path.parent
        subject_dir = class_dir.parent
        class_name = class_dir.name or None
        subject = subject_dir.name if subject_dir != class_dir else None
        return subject, class_name

    @staticmethod
    def _safe_num_pages(doc: DoclingDocument) -> int:
        try:
            return doc.num_pages()
        except Exception:  # noqa: BLE001 - defensive fallback across versions
            return len(doc.pages)

    def _extract_pages(self, doc: DoclingDocument) -> list[Page]:
        pages = []
        for page_no in sorted(doc.pages.keys()):
            page_item = doc.pages[page_no]
            size = getattr(page_item, "size", None)
            pages.append(
                Page(
                    page_number=page_no,
                    width=getattr(size, "width", None),
                    height=getattr(size, "height", None),
                )
            )
        logger.info("Pages processed: %d", len(pages))
        return pages

    # ------------------------------------------------------------------
    # Structure (sections, reading order, hierarchy)
    # ------------------------------------------------------------------

    def extract_structure(self, doc: DoclingDocument) -> tuple[str | None, list[Section], list[str]]:
        """Walk the document in reading order and build the section tree.

        Returns `(title, top_level_sections, warnings)`. Tables and images
        are delegated to `extract_tables`/`extract_images` for the actual
        model construction so each concern stays independently testable.
        """
        warnings: list[str] = []
        title: str | None = None
        reading_order = 0
        table_count = 0
        image_count = 0

        caption_refs = self._collect_caption_refs(doc)

        root = Section(heading=None, level=0)
        section_stack: list[Section] = [root]
        pending_list: OrderedList | BulletList | None = None

        def flush_pending_list() -> None:
            nonlocal pending_list
            if pending_list is not None:
                section_stack[-1].blocks.append(pending_list)
                pending_list = None

        for item, _tree_depth in doc.iterate_items(with_groups=False):
            page_number = item.prov[0].page_no if getattr(item, "prov", None) else None

            if isinstance(item, TitleItem):
                flush_pending_list()
                if title is None:
                    title = item.text
                continue

            if isinstance(item, SectionHeaderItem):
                flush_pending_list()
                level = max(item.level, 1)
                while len(section_stack) > 1 and section_stack[-1].level >= level:
                    section_stack.pop()
                new_section = Section(heading=item.text, level=level, page_number=page_number)
                section_stack[-1].subsections.append(new_section)
                section_stack.append(new_section)
                continue

            if isinstance(item, TableItem):
                flush_pending_list()
                table_count += 1
                table_block = self._build_table(item, doc, page_number, reading_order)
                section_stack[-1].blocks.append(table_block)
                reading_order += 1
                continue

            if isinstance(item, PictureItem):
                flush_pending_list()
                image_count += 1
                image_block = self._build_image(item, doc, page_number, reading_order)
                section_stack[-1].blocks.append(image_block)
                reading_order += 1
                continue

            if item.self_ref in caption_refs:
                # Already captured via the owning table/picture's `caption` field.
                continue

            if isinstance(item, ListItem):
                is_ordered = bool(item.enumerated)
                if (
                    pending_list is not None
                    and isinstance(pending_list, OrderedList) == is_ordered
                    and pending_list.page_number == page_number
                ):
                    pending_list.items.append(item.text)
                else:
                    flush_pending_list()
                    list_cls = OrderedList if is_ordered else BulletList
                    pending_list = list_cls(
                        items=[item.text], page_number=page_number, reading_order=reading_order
                    )
                    reading_order += 1
                continue

            flush_pending_list()

            if isinstance(item, FormulaItem):
                section_stack[-1].blocks.append(
                    Equation(text=item.text, page_number=page_number, reading_order=reading_order)
                )
                reading_order += 1
                continue

            label = getattr(item, "label", None)
            text = getattr(item, "text", None)
            if text is None:
                continue
            if label is not None and label not in _TEXT_BLOCK_LABELS:
                warnings.append(f"Unrecognized content label '{label}' treated as paragraph.")
            section_stack[-1].blocks.append(
                Paragraph(text=text, page_number=page_number, reading_order=reading_order)
            )
            reading_order += 1

        flush_pending_list()

        logger.info("Tables extracted: %d", table_count)
        logger.info("Images detected: %d", image_count)

        # Only keep the implicit root section if it actually carries content
        # (e.g. preamble text before the first heading); otherwise expose its
        # subsections directly so callers see the book's real top-level
        # chapters instead of one meaningless heading-less wrapper.
        sections = [root] if (root.blocks or root.heading is not None) else root.subsections

        return title, sections, warnings

    @staticmethod
    def _collect_caption_refs(doc: DoclingDocument) -> set[str]:
        refs: set[str] = set()
        for table in doc.tables:
            refs.update(ref.cref for ref in table.captions)
        for picture in doc.pictures:
            refs.update(ref.cref for ref in picture.captions)
        return refs

    @staticmethod
    def _resolve_caption(item: Any, doc: DoclingDocument) -> str | None:
        for ref in getattr(item, "captions", []):
            try:
                resolved = ref.resolve(doc)
            except Exception:  # noqa: BLE001 - a dangling ref shouldn't break parsing
                continue
            text = getattr(resolved, "text", None)
            if text:
                return text
        return None

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def extract_tables(self, doc: DoclingDocument) -> list[TableItem]:
        """Return the raw Docling table items found in the document."""
        return list(doc.tables)

    def _build_table(
        self, item: TableItem, doc: DoclingDocument, page_number: int | None, reading_order: int
    ) -> Table:
        try:
            grid = item.data.grid
            rows = [
                [
                    TableCell(
                        text=cell.text,
                        row_span=max(cell.row_span, 1),
                        col_span=max(cell.col_span, 1),
                        is_header=getattr(cell, "column_header", False) or getattr(cell, "row_header", False),
                    )
                    for cell in row
                ]
                for row in grid
            ]
            num_rows = item.data.num_rows
            num_cols = item.data.num_cols
        except Exception as exc:  # noqa: BLE001 - malformed table shouldn't kill the parse
            logger.warning("Failed to extract table grid on page %s: %s", page_number, exc)
            rows, num_rows, num_cols = [], 0, 0

        return Table(
            caption=self._resolve_caption(item, doc),
            rows=rows,
            num_rows=num_rows,
            num_cols=num_cols,
            page_number=page_number,
            reading_order=reading_order,
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def extract_images(self, doc: DoclingDocument) -> list[PictureItem]:
        """Return the raw Docling picture items found in the document."""
        return list(doc.pictures)

    def _build_image(
        self, item: PictureItem, doc: DoclingDocument, page_number: int | None, reading_order: int
    ) -> ImageReference:
        bbox = item.prov[0].bbox if item.prov else None
        width = (bbox.r - bbox.l) if bbox is not None else None
        height = (bbox.t - bbox.b) if bbox is not None else None
        return ImageReference(
            reference_id=item.self_ref,
            caption=self._resolve_caption(item, doc),
            page_number=page_number,
            reading_order=reading_order,
            width=width,
            height=height,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_document(self, document: Document) -> None:
        """Raise `DocumentValidationError` if a parsed document is unusable."""
        if document.metadata.total_pages == 0:
            raise DocumentValidationError(
                f"Document has zero pages: {document.metadata.source_path}"
            )
        has_content = any(_section_has_content(section) for section in document.sections)
        if not has_content:
            raise EmptyDocumentError(
                f"No extractable content found in: {document.metadata.source_path}"
            )


def _section_has_content(section: Section) -> bool:
    if section.blocks:
        return True
    return any(_section_has_content(sub) for sub in section.subsections)
