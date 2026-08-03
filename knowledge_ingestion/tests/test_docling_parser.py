"""Unit tests for DoclingParser — a real DoclingDocument built via
docling-core's own builder API (not the heavy, torch-backed `docling`
conversion package) driven through a fake converter, so no PDF conversion
or ML model is actually needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel

from models.document import BulletList, Equation, ImageReference, OrderedList, Paragraph, Table
from parser.docling_parser import DoclingParser
from parser.exceptions import EmptyDocumentError, UnsupportedDocumentError


def _prov(page_no: int = 1) -> ProvenanceItem:
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(l=0, t=100, r=100, b=0, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 10),
    )


def _build_fake_document() -> DoclingDocument:
    doc = DoclingDocument(name="chapter1")
    doc.add_page(page_no=1, size=Size(width=595.0, height=842.0))
    doc.add_page(page_no=2, size=Size(width=595.0, height=842.0))

    doc.add_title("Introduction to Cell Biology", prov=_prov(1))

    h1 = doc.add_heading("Chapter 1: The Cell", level=1, prov=_prov(1))
    doc.add_text(DocItemLabel.PARAGRAPH, "The cell is the basic unit of life.", parent=h1, prov=_prov(1))

    h2 = doc.add_heading("1.1 Cell Types", level=2, parent=h1, prov=_prov(1))
    doc.add_text(DocItemLabel.PARAGRAPH, "There are two main types of cells.", parent=h2, prov=_prov(1))

    ordered = doc.add_list_group(parent=h2)
    doc.add_list_item("Prokaryotic cells", enumerated=True, parent=ordered, prov=_prov(1))
    doc.add_list_item("Eukaryotic cells", enumerated=True, parent=ordered, prov=_prov(1))

    bullet = doc.add_list_group(parent=h2)
    doc.add_list_item("No nucleus", enumerated=False, parent=bullet, prov=_prov(1))
    doc.add_list_item("Simpler structure", enumerated=False, parent=bullet, prov=_prov(1))

    doc.add_formula("E = mc^2", parent=h2, prov=_prov(1))

    table_data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            TableCell(text="Feature", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
            TableCell(text="Prokaryote", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, column_header=True),
            TableCell(text="Nucleus", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1),
            TableCell(text="Absent", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2),
        ],
    )
    caption = doc.add_text(DocItemLabel.CAPTION, "Table 1: Cell comparison", prov=_prov(2))
    doc.add_table(data=table_data, parent=h2, prov=_prov(2), caption=caption)

    image_caption = doc.add_text(DocItemLabel.CAPTION, "Figure 1: A cell diagram", prov=_prov(2))
    doc.add_picture(parent=h2, prov=_prov(2), caption=image_caption)

    h3 = doc.add_heading("Chapter 2: Genetics", level=1, prov=_prov(2))
    doc.add_text(DocItemLabel.PARAGRAPH, "Genetics is the study of heredity.", parent=h3, prov=_prov(2))

    return doc


class _FakeStatus:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeResult:
    def __init__(self, document: DoclingDocument) -> None:
        self.status = _FakeStatus("SUCCESS")
        self.document = document
        self.errors: list = []


class _FakeConverter:
    def __init__(self, document: DoclingDocument) -> None:
        self._document = document

    def convert(self, pdf_path):
        return _FakeResult(self._document)


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "downloads" / "biology" / "class11" / "chapter1.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake content for existence/size checks")
    return pdf_path


@pytest.fixture
def parser() -> DoclingParser:
    return DoclingParser(converter=_FakeConverter(_build_fake_document()))


def test_parse_infers_subject_and_class_from_path(parser: DoclingParser, sample_pdf_path: Path):
    document = parser.parse(sample_pdf_path)

    assert document.title == "Introduction to Cell Biology"
    assert document.metadata.subject == "biology"
    assert document.metadata.class_name == "class11"
    assert document.metadata.book_name == "chapter1"
    assert document.metadata.total_pages == 2
    assert len(document.pages) == 2


def test_explicit_subject_class_override_beats_path_guessing(parser: DoclingParser, tmp_path: Path):
    # Regression guard: a flat S3 layout (<subject>/<book>.pdf, no class
    # subfolder — e.g. "Accounts/10Cash Flow Statement.pdf") downloads to a
    # local path that is structurally indistinguishable from a real
    # <subject>/<class>/<book>.pdf layout. Path-based guessing alone cannot
    # tell them apart and gets it wrong; the caller (IngestionPipeline)
    # knows the true values from S3Loader and must be able to force them.
    flat_pdf_path = tmp_path / "downloads" / "Accounts" / "10Cash Flow Statement.pdf"
    flat_pdf_path.parent.mkdir(parents=True)
    flat_pdf_path.write_bytes(b"%PDF-1.4 fake")

    guessed = parser.parse(flat_pdf_path)
    assert guessed.metadata.subject == "downloads"  # the bug, if nothing overrides it
    assert guessed.metadata.class_name == "Accounts"

    corrected = parser.parse(flat_pdf_path, subject="Accounts", class_name=None)
    assert corrected.metadata.subject == "Accounts"
    assert corrected.metadata.class_name is None


def test_parse_builds_section_hierarchy_without_spurious_wrapper(parser: DoclingParser, sample_pdf_path: Path):
    document = parser.parse(sample_pdf_path)

    assert [s.heading for s in document.sections] == ["Chapter 1: The Cell", "Chapter 2: Genetics"]
    chapter1 = document.sections[0]
    assert len(chapter1.subsections) == 1
    assert chapter1.subsections[0].heading == "1.1 Cell Types"


def test_parse_preserves_mixed_content_in_reading_order(parser: DoclingParser, sample_pdf_path: Path):
    document = parser.parse(sample_pdf_path)
    subsection = document.sections[0].subsections[0]

    block_types = [type(b).__name__ for b in subsection.blocks]
    assert block_types == [
        "Paragraph", "OrderedList", "BulletList", "Equation", "Table", "ImageReference",
    ]

    ordered = subsection.blocks[1]
    assert isinstance(ordered, OrderedList)
    assert ordered.items == ["Prokaryotic cells", "Eukaryotic cells"]

    bullet = subsection.blocks[2]
    assert isinstance(bullet, BulletList)
    assert bullet.items == ["No nucleus", "Simpler structure"]

    equation = subsection.blocks[3]
    assert isinstance(equation, Equation)
    assert equation.text == "E = mc^2"

    table = subsection.blocks[4]
    assert isinstance(table, Table)
    assert table.caption == "Table 1: Cell comparison"
    assert table.rows[0][0].text == "Feature" and table.rows[0][0].is_header
    assert table.rows[1][1].text == "Absent"

    image = subsection.blocks[5]
    assert isinstance(image, ImageReference)
    assert image.caption == "Figure 1: A cell diagram"


def test_reading_order_is_globally_monotonic_and_unique(parser: DoclingParser, sample_pdf_path: Path):
    document = parser.parse(sample_pdf_path)
    orders: list[int] = []

    def collect(section):
        orders.extend(b.reading_order for b in section.blocks)
        for sub in section.subsections:
            collect(sub)

    for section in document.sections:
        collect(section)

    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_empty_file_is_rejected_before_conversion(parser: DoclingParser, tmp_path: Path):
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    with pytest.raises(EmptyDocumentError):
        parser.parse(empty_pdf)


def test_non_pdf_extension_is_rejected(parser: DoclingParser, tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello")

    with pytest.raises(UnsupportedDocumentError):
        parser.parse(text_file)


@pytest.mark.integration
def test_parser_config_do_ocr_propagates_to_real_docling_pipeline():
    # Needs the full `docling` package (not just `docling-core`), since it
    # inspects a real DocumentConverter's wired-up PdfPipelineOptions -
    # marked @integration so the fast/offline suite doesn't require it.
    from docling.datamodel.base_models import InputFormat

    from models.document import ParserConfig

    default_config = ParserConfig()
    assert default_config.do_ocr is True  # matches Docling's own default; no behavior change for existing users

    light_config = ParserConfig(do_ocr=False, do_table_structure=True)
    converter = DoclingParser._build_default_converter(light_config)
    pdf_options = converter.format_to_options[InputFormat.PDF].pipeline_options

    assert pdf_options.do_ocr is False
    assert pdf_options.do_table_structure is True
