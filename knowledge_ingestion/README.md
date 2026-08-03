# Knowledge Ingestion Pipeline (YoTutor)

## Purpose

This repository converts educational PDFs into searchable vector embeddings
for **YoTutor**, an AI tutoring platform, and retrieves the most relevant
chunks for a given teaching moment:

```
AWS S3 → Download PDFs → Parse PDFs → Chunk Documents → Generate Embeddings → Store in Qdrant → Retrieve Relevant Chunks
```

This repository does **not** handle authentication, MongoDB, APIs, iOS,
frontend, LLM responses/chat, tutor orchestration, voice, or animation.
Those concerns live in other services — this repo hands them ranked chunks
of textbook content, nothing more.

## Architecture

The codebase follows Clean Architecture principles: each module owns exactly
one responsibility, depends only on configuration (never on other pipeline
stages directly), and is wired together at a single composition root
(`main.py`). This keeps every stage independently testable and swappable —
for example, `VectorStore` could be replaced with an alternate vector
database without touching parsing, chunking, or embedding code.

```
S3Loader  →  DoclingParser  →  LlamaChunker  →  EmbeddingService  →  VectorStore
   │               │                │                 │                  │
   └───────────────┴────────────────┴─────────────────┴──────────────────┘
                              orchestrated by
                          pipeline/ingest_pipeline.py
                       (IngestionPipeline, constructed in main.py)

                                                        VectorStore  ◄──────┐
                                                             ▲              │
                                                             │              │
                                                        Retriever  ◄──  EmbeddingService
                                                     (query-time reader,
                                                    not part of IngestionPipeline)
```

Each ingestion stage is injected as a dependency into `IngestionPipeline`,
so components can be mocked in tests or swapped for alternate
implementations later without modifying the orchestration logic.
`Retriever` is a separate, query-time consumer of the same
`VectorStore`/`EmbeddingService` — it reads what `IngestionPipeline` has
written, but isn't itself part of the ingestion pipeline or its
composition root.

## Folder Structure

```
knowledge_ingestion/
├── config/                 # Settings and logging — shared by every module
│   ├── settings.py         # Pydantic Settings, loaded from .env
│   └── logging.py          # Centralized Rich-based logger factory
├── loaders/                 # S3 → local filesystem
│   ├── s3_loader.py         # S3Loader — discovery + download (implemented)
│   └── exceptions.py        # Custom exceptions for the S3 loading stage
├── parser/                  # PDF → structured document (Docling)
│   ├── docling_parser.py    # DoclingParser — parsing + structure extraction (implemented)
│   └── exceptions.py        # Custom exceptions for the parsing stage
├── chunking/                 # Document → retrieval-sized chunks (LlamaIndex)
│   ├── llama_chunker.py      # LlamaChunker — education-aware chunking (implemented)
│   └── exceptions.py         # Custom exceptions for the chunking stage
├── embedding/                # Text chunks → vectors (BAAI/bge-m3)
│   ├── bge_embedder.py       # EmbeddingService — batched, validated embedding (implemented)
│   └── exceptions.py         # Custom exceptions for the embedding stage
├── vectorstore/               # Vectors → Qdrant collection
│   ├── qdrant_client.py       # CollectionManager/BatchUploader/VectorStore (implemented)
│   └── exceptions.py          # Custom exceptions for the vector-store stage
├── retrieval/                 # TeachingContext → ranked RetrievedChunks
│   ├── retriever.py           # Retriever — context-aware search (implemented)
│   └── exceptions.py          # Custom exceptions for the retrieval stage
├── pipeline/                  # Orchestrates all ingestion stages end to end
│   ├── ingest_pipeline.py     # IngestionPipeline — checkpointed, resumable orchestration (implemented)
│   └── exceptions.py          # Custom exceptions for the orchestrator
├── models/                    # Shared Pydantic data models (domain types)
│   ├── s3_object.py           # S3PdfObject — a discovered PDF's identity/metadata
│   ├── document.py            # Document/Section/Paragraph/Table/... — parsed document tree
│   ├── chunk.py                # Chunk/ChunkMetadata/ChunkingConfig — chunker input/output
│   ├── embedded_chunk.py       # EmbeddedChunk/EmbeddingConfig — embedder input/output
│   ├── qdrant.py                # QdrantConfig/UploadResult/CollectionStats — vector-store I/O
│   ├── retrieval.py             # TeachingContext/RetrievalConfig/RetrievedChunk — retrieval I/O
│   └── pipeline.py               # PipelineConfig/PipelineState/PipelineStatistics — orchestrator I/O
├── utils/                     # Cross-cutting helper utilities
│   └── retry.py                # Generic exponential-backoff retry helper
├── tests/                     # Unit and integration tests
├── main.py                    # Composition root / entrypoint
├── pyproject.toml             # Poetry configuration and dependencies
├── .env.example                # Template for required environment variables
└── README.md
```

## Installation

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Running with Poetry

```bash
cp .env.example .env
# fill in .env with real AWS / Qdrant credentials

poetry run python main.py
```

## Environment Variables

| Variable                | Description                                   |
|--------------------------|------------------------------------------------|
| `AWS_ACCESS_KEY_ID`     | AWS access key with S3 read permissions        |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key                          |
| `AWS_REGION`            | AWS region hosting the S3 bucket               |
| `S3_BUCKET_NAME`        | Bucket containing source PDFs                  |
| `QDRANT_URL`            | Qdrant instance endpoint                       |
| `QDRANT_API_KEY`        | Qdrant API key                                 |
| `QDRANT_COLLECTION`     | Target Qdrant collection name                  |
| `EMBEDDING_MODEL`       | HuggingFace embedding model (default: `BAAI/bge-m3`) |
| `LOG_LEVEL`             | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, ...) |

`AWS_SECRET_ACCESS_KEY` and `QDRANT_API_KEY` are loaded as Pydantic
`SecretStr`, not plain `str` — they render as `SecretStr('**********')` in
any log line, traceback, or `repr()` that touches `Settings`, and the real
value is only reachable via `.get_secret_value()` at the one or two call
sites that actually need to hand it to boto3/qdrant-client. `.env` usage is
unchanged; this only affects how the loaded value behaves in-process.

## Testing

```bash
poetry install --with dev
pytest                        # full suite (57 tests)
pytest -m "not integration"   # fast, fully offline subset (~55 tests, no network/model download)
pytest --cov                  # with coverage (pytest-cov is a dev dependency)
```

Every module has a corresponding `tests/test_*.py` file. Most tests are
fast, hermetic unit tests built the same way each module's own DI supports
being tested: a mocked boto3 client for `S3Loader`, a real `DoclingDocument`
built via `docling-core`'s own builder API (not the heavy `docling`
conversion package) for `DoclingParser`, a fake/deterministic embedding
model for `EmbeddingService`/`Retriever`, qdrant-client's real in-memory
mode (`QdrantClient(":memory:")`) for `VectorStore`/`Retriever`, and fake
collaborators duck-typing the real stage interfaces for `IngestionPipeline`.
A handful of `@pytest.mark.integration` tests use a real small
sentence-transformers model for genuine end-to-end confidence (including a
semantic-relevance check: "What is the powerhouse of the cell?" correctly
retrieves the mitochondria chunk over the plain cell-definition chunk) —
these need network access on first run to download the model, so they're
excluded from the fast subset.

## S3 Loader

`loaders/s3_loader.py` implements `S3Loader`, the only stage currently
implemented in this repository. Its sole job is: **connect to S3, discover
what's there, and download it locally while preserving structure.** It does
not parse, chunk, embed, or write to Qdrant.

### Subject discovery — no hardcoded subject names

The bucket is treated as a flat list of first-level "folders" (S3 prefixes),
discovered at runtime via `list_objects_v2` with `Delimiter="/"`:

```python
loader = S3Loader()
subjects = loader.discover_subjects()   # e.g. ["biology", "physics", ...]
```

No subject name (`biology`, `physics`, ...) appears anywhere in the code.
If a new prefix (e.g. `computer_science/`) is uploaded to the bucket
tomorrow, the very next call to `discover_subjects()` returns it — nothing
in `loaders/s3_loader.py` needs to change.

The same pattern applies one level deeper for classes:

```python
classes = loader.discover_classes("biology")   # e.g. ["class11", "class12"]
```

...and recursively for PDFs nested at any depth under a subject:

```python
pdfs = loader.discover_pdf_files("biology")     # list[S3PdfObject]
all_pdfs = loader.list_all_pdfs()               # every PDF, every subject
```

Each `S3PdfObject` (`models/s3_object.py`) carries `subject` and
`class_name`, both parsed from the object key's own path segments — never
looked up against a fixed list.

### Folder hierarchy

Local downloads mirror the S3 key structure exactly, rooted at
`download_root` (defaults to `downloads/`):

```
S3 key: biology/class11/chapter1.pdf
     → downloads/biology/class11/chapter1.pdf
```

Downloads are atomic: each file is written to a `.part` sibling and only
renamed into place once complete, so a crash mid-download never leaves a
corrupt/partial file at the final path.

### Downloading

```python
loader.download_subject("biology")   # every PDF under one subject
loader.download_all()                # every PDF, every subject
loader.download_file(pdf_object)     # a single S3PdfObject
```

- Files that already exist locally are **skipped** (no S3 API call) unless
  `overwrite=True` is passed.
- `download_subject` / `download_all` parallelize downloads with a
  `ThreadPoolExecutor` (`max_workers`, default 8) and show a `tqdm`
  progress bar.
- Discovery results (`discover_subjects`, `discover_classes`,
  `discover_pdf_files`) are cached on the instance to avoid redundant S3
  listing calls; pass `use_cache=False` or call `loader.clear_cache()` to
  force a fresh listing.

### Error handling

All failures surface as typed exceptions from `loaders/exceptions.py`
(`S3AuthenticationError`, `S3AccessDeniedError`, `BucketNotFoundError`,
`ObjectNotFoundError`, `S3ConnectionError`, `S3DownloadError`) instead of
raw `botocore` exceptions, so callers don't need to know boto3's error
shapes. Transient failures (timeouts, 5xx, throttling) are retried with
exponential backoff (`utils/retry.py`) before being raised.

### Dependency injection / testability

`S3Loader` accepts an optional `s3_client` in its constructor — pass a
mocked/stubbed boto3 client in tests instead of talking to real AWS. The
loader holds no module-level state; everything (caches, client, bucket
name) lives on the instance.

## PDF Parser

`parser/docling_parser.py` implements `DoclingParser`. Its sole job is:
**take one PDF (already downloaded by `S3Loader`) and turn it into a
structured `Document` object.** It does not chunk, embed, write to Qdrant,
or know that those stages exist — `parser/` has zero imports from
`chunking/`, `embedding/`, or `vectorstore/`.

### Why Docling

Docling was chosen because it produces a genuine document *model* —
headings, paragraphs, lists, tables (as grids, not flattened text),
pictures, and formulas, each tagged with a semantic label and page
provenance — rather than a bag of loose text lines like most PDF-to-text
libraries. That structure is exactly what a chunking stage needs later to
make smart splitting decisions (e.g. "never split a table row" or "keep a
heading with its first paragraph") instead of chunking blindly on
character count.

### How parsing works

```python
parser = DoclingParser()
document = parser.parse("downloads/biology/class11/chapter1.pdf")
```

1. **Validate the file** — must exist, have a `.pdf` extension, and be
   non-empty — before ever invoking Docling.
2. **Convert** via Docling's `DocumentConverter` into a `DoclingDocument`.
   Conversion failures and `FAILURE`-status results are mapped to
   `CorruptedPDFError` or `PasswordProtectedPDFError`; a `PARTIAL_SUCCESS`
   status is logged as a warning and parsing continues with whatever
   content Docling did recover.
3. **Walk the document in reading order** via `DoclingDocument.iterate_items()`
   and build a `Section` tree: `SectionHeaderItem`s open/close sections by
   heading level, and every other item becomes a typed `ContentBlock`
   (`Paragraph`, `OrderedList`, `BulletList`, `Table`, `ImageReference`,
   `Equation`) appended to the currently-open section — never flattened to
   a single blob of text.
4. **Validate the result** — zero pages or zero extractable content raises
   `EmptyDocumentError`/`DocumentValidationError` rather than silently
   returning an empty document.

`parse_directory(directory)` is a thin generator over `parse()` for every
`*.pdf` found recursively under a directory: one file failing (corrupted,
password-protected, ...) is logged and skipped rather than aborting the
whole batch, while still processing one document at a time (no bulk
in-memory loading — a future batch/parallel stage can build on top of this
without changing `DoclingParser` itself).

### What information is preserved

| Preserved | How |
|---|---|
| Title | `Document.title`, from Docling's `TitleItem` |
| Headings (all levels) | `Section.heading` / `Section.level`, nested via `Section.subsections` |
| Paragraphs | `Paragraph` blocks |
| Ordered / bullet lists | `OrderedList` / `BulletList`, distinguished via Docling's `ListItem.enumerated` flag, consecutive items of the same kind grouped into one block |
| Tables | `Table.rows` — a row-major grid of `TableCell` (with row/col span and header flag), never flattened to text |
| Captions | Resolved onto the owning `Table.caption` / `ImageReference.caption`; an unlinked caption falls back to a `Paragraph` so it's never dropped |
| Images | `ImageReference` — Docling's internal `reference_id` + page/caption/size only, **never the image bytes** |
| Page numbers | Every block/section carries `page_number`, taken from Docling's provenance data |
| Reading order | Every block has a document-wide, monotonically increasing `reading_order` — the chunking module can flatten and sort across sections without reconstructing tree traversal itself |
| Equations | `Equation` blocks, from Docling's `FormulaItem` |
| Section hierarchy | The `Section.subsections` tree, built from heading levels during the reading-order walk |

Anything Docling labels that doesn't map to one of the above (e.g. page
headers/footers are excluded by Docling itself; unrecognized labels) still
becomes a `Paragraph` rather than being silently dropped, with a note added
to `Document.warnings`.

### Output data model (`models/document.py`)

```
Document
├── metadata: Metadata        (document_id, subject, class_name, book_name,
│                               source_path, total_pages, parser_version,
│                               parsed_at)
├── title: str | None
├── pages: list[Page]         (page_number, width, height)
├── sections: list[Section]   (heading, level, page_number,
│     │                        blocks: list[ContentBlock], subsections: list[Section])
│     └── subsections: list[Section]   (recursive)
└── warnings: list[str]
```

`ContentBlock` is a Pydantic discriminated union of `Paragraph`,
`OrderedList`, `BulletList`, `Table`, `ImageReference`, and `Equation`,
tagged by a `type` field — so a consumer can `match`/branch on block type
without `isinstance` checks, and (de)serializes losslessly through
`model_dump()` / `model_validate()`.

### Metadata + subject/class

`parse()` accepts optional `subject`/`class_name` keyword arguments.
`IngestionPipeline` always passes them explicitly, using the exact values
`S3Loader` already discovered from the real S3 key (`S3PdfObject.subject`/
`class_name`) — the reliable source of truth, since it comes straight from
the bucket's own folder structure, including layouts with no class-level
subfolder at all (`<subject>/<book>.pdf`, `class_name` simply empty).

When neither is given (standalone/direct use, e.g. `parse_directory()`),
`parse()` falls back to guessing from the local file's own path — the
immediate parent directory is treated as the class, and its parent as the
subject (`.../biology/class11/chapter1.pdf` → `subject="biology"`,
`class_name="class11"`). This fallback is inherently a guess: a flat
`<subject>/<book>.pdf` S3 layout downloads to a local path that's
structurally identical to a real `<subject>/<class>/<book>.pdf` one, so
path shape alone can't distinguish them — passing `subject`/`class_name`
explicitly whenever the true values are known (as `IngestionPipeline`
does) is the reliable path; the fallback exists only for when they aren't.

### Error handling

`parser/exceptions.py` defines `CorruptedPDFError`,
`PasswordProtectedPDFError`, `UnsupportedDocumentError`,
`EmptyDocumentError`, and `DocumentValidationError` — all callers need to
handle is this small set, not Docling/PDF-backend internals.

### Dependency injection / testability

`DoclingParser` accepts an optional `converter` in its constructor; the
real `docling.document_converter.DocumentConverter` is only imported lazily
on first use if none is injected, so this module — and its structure
extraction logic — stays importable and unit-testable even without the
full (torch-backed) `docling` package installed, using a stub converter or
a `DoclingDocument` built directly via `docling-core`'s own builder API.

## Chunking

`chunking/llama_chunker.py` implements `LlamaChunker`. Its sole job is:
**take one parsed `Document` (from `DoclingParser`) and turn it into a
`list[Chunk]` ready for embedding.** It does not embed, write to Qdrant, or
know that retrieval/LLMs exist — `chunking/` has zero imports from
`embedding/` or `vectorstore/`.

### Why not just fixed-size / default chunking

Splitting purely on a token count (LlamaIndex's default `SentenceSplitter`
behavior) has no idea that a paragraph is a *definition*, that four
consecutive lines are one *table row*, or that a formula and the sentence
explaining it belong together. Naive fixed-size splitting will cut a table
in half, split a solved example after step 2 of 5, or separate a formula
from the paragraph that explains it — each of those breaks the chunk's
standalone meaning, which directly hurts retrieval (see below). So
`LlamaChunker` uses `SentenceSplitter` as the *base* splitter for ordinary
paragraph text — it's good at sentence/paragraph-aware splitting — but adds
an education-aware layer in front of it that decides, block by block,
whether a piece of content is even allowed to reach the splitter.

### How chunking works

```python
chunker = LlamaChunker(config=ChunkingConfig(target_chunk_size_tokens=512, chunk_overlap_tokens=80))
chunks = chunker.chunk(document)   # list[Chunk]
```

1. **Walk the `Section` tree** (depth-first, same order the parser built
   it) so `chapter` (the top-level section heading) and `section` (the
   nearest heading, which may be a subsection) are tracked per block
   without any string-guessing of heading text.
2. **Group each section's blocks into units** — this is the education-aware
   step:
   - `Table` → rendered as a caption + `" | "`-joined grid, always **atomic**.
   - `ImageReference` → replaced with a `[IMAGE: <caption>]` placeholder,
     always atomic, with `reference_id` preserved in `ChunkMetadata.image_references`.
   - `Equation` → merged with the paragraph immediately after it (or, if
     there isn't one, the paragraph immediately before it) into one atomic
     "formula + explanation" unit, so a formula is never separated from
     what explains it.
   - `Paragraph` → matched against patterns for `Definition:`, `Example
     N:` / `Solved Example:`, `Theorem:` / `Lemma:` / `Corollary:`, and
     `Note:` / `Important Note:`. A match makes the whole paragraph
     **atomic — kept whole no matter how large**; everything else is
     ordinary **splittable** text.
   - `OrderedList` / `BulletList` → rendered as numbered/bulleted text and
     treated as splittable (not explicitly protected in the spec, but kept
     as one block by default and only broken up by the splitter if it's
     genuinely too long).
3. **Merge consecutive splittable units and hand them to `SentenceSplitter`**
   (`chunk_size`/`chunk_overlap` from `ChunkingConfig`, `paragraph_separator="\n\n"`
   so it respects the paragraph joins). Atomic units bypass the splitter
   entirely and become their own single chunk. A section boundary always
   ends a merge run, so a chunk never spans two different sections.
4. **Prefix each chunk with its heading breadcrumb** (`[Subject > Chapter >
   Section]`) when `include_heading_context=True` (default) — see "why
   this improves retrieval" below.
5. **Assign sequence-wide metadata** (`chunk_number`, `total_chunks`) once
   every section has been processed, and **validate** the result (see
   Validation below) before returning.

### Chunk metadata

Every `Chunk` carries a `ChunkMetadata` (`models/chunk.py`) with exactly
the fields needed to become a Qdrant payload later, nothing dropped along
the way:

`chunk_id`, `document_id`, `subject`, `class_name`, `chapter`, `section`,
`page_number` (first page) + `page_numbers` (full span — a split
paragraph run is attributed the page range it was merged from, since exact
per-sentence page attribution isn't available at paragraph granularity),
`source_pdf`, `parser_version`, `chunk_number`, `total_chunks`,
`character_count`, `estimated_token_count`, `content_kind` (`text` /
`definition` / `example` / `theorem` / `note` / `table` / `formula` /
`image`), `image_references`.

### Configuration (`ChunkingConfig`)

| Field | Default | Meaning |
|---|---|---|
| `target_chunk_size_tokens` | `512` | Target size passed to `SentenceSplitter` |
| `chunk_overlap_tokens` | `80` | Overlap passed to `SentenceSplitter`; must be `<` target size (validated) |
| `min_chunk_size_tokens` | `20` | Below this, a chunk is flagged (not rejected) during validation |
| `include_heading_context` | `True` | Prefix each chunk's text with its `[Subject > Chapter > Section]` breadcrumb |

Token counts use LlamaIndex's default tokenizer (`llama_index.core.utils.get_tokenizer()`,
tiktoken-based); a custom `tokenizer` callable can be injected into
`LlamaChunker(tokenizer=...)` for tests or an alternate model's tokenizer.

### Validation

`chunk()` calls `validate_chunks()` automatically and raises
`ChunkValidationError` (`chunking/exceptions.py`) for structural defects —
empty chunk text or missing required metadata. Duplicate chunks and chunks
outside the configured size range are logged as warnings, not hard
failures, since (for example) a legitimately oversized "never split"
definition is expected, not a bug.

### Dependency injection / testability

`LlamaChunker` takes `config: ChunkingConfig` and `tokenizer` as
constructor arguments and holds no other mutable state, so it can chunk
many documents sequentially, or be instantiated per-worker for future
parallel/batch processing, without any shared state between calls.

## Embedding

`embedding/bge_embedder.py` implements `EmbeddingService`. Its sole job is:
**take a `list[Chunk]` (from `LlamaChunker`) and turn it into a
`list[EmbeddedChunk]`** — a vector plus a Qdrant-ready payload per chunk. It
does not write to Qdrant, retrieve anything, or know an LLM exists —
`embedding/` has zero imports from `vectorstore/`.

### Why BAAI/bge-m3

`BAAI/bge-m3` is a strong multilingual, multi-granularity embedding model
that performs well on both short queries and longer passages — a good fit
for a tutor platform that needs to match short student questions against
paragraph/section-length textbook chunks, potentially across languages.
Loaded locally via `sentence-transformers` (no external embedding API call
per chunk), which keeps ingestion self-contained and avoids per-token API
costs at "millions of pages" scale.

### Embedding workflow

```python
service = EmbeddingService()   # model_name defaults from Settings.EMBEDDING_MODEL
embedded_chunks = service.embed_chunks(chunks)   # list[EmbeddedChunk]
```

1. **Reject empty input up front** — any chunk with empty/whitespace-only
   text raises `EmptyChunkTextError` immediately (fail-fast) rather than
   silently embedding meaningless text or silently dropping a chunk, which
   would break the "embedding count matches chunk count" guarantee.
2. **Load the model once, lazily** — on first use, not at construction. The
   loaded `SentenceTransformer` and its embedding dimension are cached on
   the instance for the rest of its lifetime; a real model download only
   happens once no matter how many documents are embedded through the same
   `EmbeddingService` instance. A pre-built model can be injected via
   `EmbeddingService(model=...)`, which is how this module is unit-tested
   without downloading the real (multi-GB) `BAAI/bge-m3`.
3. **Pick a device automatically**: CUDA if available, then Apple MPS, then
   CPU — or an explicit `EmbeddingConfig.device` override, which itself
   falls back to CPU with a warning if the requested device turns out to be
   unavailable.
4. **Encode in configurable batches** (`EmbeddingConfig.batch_size`,
   default `32`) — chunk text is embedded batch by batch (logged as
   "Processing batch N/M") rather than one giant call, bounding peak
   memory/GPU usage predictably for very large textbooks.
5. **Normalize and validate every vector** — `normalize_embeddings=True` by
   default (cosine similarity becomes a plain dot product downstream in
   Qdrant); each vector is checked for the expected dimension and for
   NaN/Inf values, raising `InvalidVectorError` rather than silently
   shipping a broken vector into the index.
6. **Assemble `EmbeddedChunk`s and validate the batch** —
   `validate_embedded_chunks()` runs automatically at the end of
   `embed_chunks()`, checking embedding count vs. chunk count, no missing
   vectors, correct dimensions, and that `chunk_id`/`document_id` still
   line up 1:1 with the input; raises `EmbeddingValidationError` on any
   mismatch.

### Batch processing

| Field (`EmbeddingConfig`) | Default | Meaning |
|---|---|---|
| `model_name` | from `Settings.EMBEDDING_MODEL` | Swapping models is a `.env` change, not a code change |
| `batch_size` | `32` | Chunks per `model.encode()` call |
| `normalize_embeddings` | `True` | L2-normalize every vector before returning |
| `device` | `None` (auto) | `"cuda"` / `"mps"` / `"cpu"`, or `None` to auto-detect |
| `max_retries` | `3` | Retries for *model loading* only (network-bound HF Hub download) — reuses `utils/retry.py` |
| `encode_timeout_seconds` | `None` (disabled) | Optional wall-clock timeout per batch; raises `EmbeddingTimeoutError` |

### Performance considerations

- **Model loaded once**, not per document or per batch — verified by
  reusing one `EmbeddingService` instance across multiple `embed_chunks()`
  calls in testing; only the first call pays the load cost.
- **GPU used automatically when available**, with a safe CPU fallback —
  this repository's dev sandbox has no CUDA device, so both paths were
  exercised: automatic CPU selection when nothing is available, and a
  warning-and-fallback when a device is explicitly requested but absent.
- **Stateless per call** — `EmbeddingService` holds only its config and
  cached model, never per-document state, so a future orchestration stage
  can safely instantiate one `EmbeddingService` per worker for parallel
  batch embedding without any shared mutable state.
- Retries are deliberately applied to **model loading only**, not to
  `encode()` calls — a transient network blip fetching model weights
  benefits from a retry, but retrying a CUDA OOM or a shape error on the
  same batch would just fail identically again.

### Every `EmbeddedChunk` contains

`vector`, `chunk_id`, `document_id`, `subject`, `class_name`, `chapter`,
`section`, `page_number`, `source_pdf`, `chunk_number`, `total_chunks`,
`token_count`, `character_count`, `original_text`, `metadata` (the full
original `ChunkMetadata`, so nothing from the chunking stage is lost even
if not promoted to a top-level field), `embedding_model`,
`embedding_timestamp`.

### Error handling

`embedding/exceptions.py` defines `ModelLoadError`, `EmptyChunkTextError`,
`InvalidVectorError`, `EmbeddingGenerationError`, `EmbeddingTimeoutError`,
and `EmbeddingValidationError` — all callers need to handle is this small
set, not sentence-transformers/torch/HTTP internals.

## Vector Store

`vectorstore/qdrant_client.py` implements three classes: `CollectionManager`
(collection lifecycle), `BatchUploader` (chunked point construction +
upload), and `VectorStore` (the facade a caller actually uses, composing
the other two). Its sole job is: **take a `list[EmbeddedChunk]` (from
`EmbeddingService`) and get it durably stored in Qdrant.** It does not do
semantic search, retrieval, or orchestration — `vectorstore/` has zero
imports from a retriever, an LLM, or `pipeline/`.

```python
store = VectorStore()   # url/api_key default from Settings
results = store.upsert_vectors(embedded_chunks)   # list[UploadResult]
```

### Collection strategy — one collection per subject, never hardcoded

The collection a chunk is written to is always `chunk.subject` itself —
`biology` chunks go to a `biology` collection, `physics` chunks to a
`physics` collection. No subject name is ever hardcoded: `upsert_vectors`
partitions the input list by each chunk's own `subject` field, and
`CollectionManager.create_collection` is called (idempotently — a no-op if
the collection already exists) the first time a subject is seen. Upload a
batch containing a `computer_science` chunk tomorrow, with zero code
changes, and a `computer_science` collection is created automatically the
same way `biology` was today — verified directly in testing by uploading a
never-before-seen subject and confirming its collection appeared with no
code change. Collection vector size is likewise never hardcoded: it's read
straight off the actual vectors in the batch being uploaded (`len(vector)`),
not assumed from a specific embedding model's known dimension — so this
module doesn't need to know or guess what `EmbeddingConfig.model_name` was.
Distance is fixed to **Cosine**, per the embedding stage's normalized
vectors.

### Payload structure

Every point's payload matches the spec's example exactly for the shared
keys, plus two additional fields present on `EmbeddedChunk` that the
example didn't list but "store ALL metadata" requires keeping:

```json
{
  "chunk_id": "...", "document_id": "...", "subject": "...", "class": "...",
  "chapter": "...", "section": "...", "page": 42, "source_pdf": "...",
  "chunk_number": 8, "total_chunks": 25, "token_count": 486,
  "embedding_model": "BAAI/bge-m3", "text": "...", "metadata": {...},
  "character_count": 1834, "embedding_timestamp": "2026-07-30T12:00:00Z"
}
```

`metadata` is the full original `ChunkMetadata` dict carried over from the
chunking stage unchanged, so anything not promoted to a top-level payload
key (e.g. `content_kind`, `page_numbers`, `parser_version`) still survives.
Payload indexes are created automatically on `document_id`, `class`,
`chapter`, `section`, and `page` — the fields worth filtering on within a
subject's collection (`subject` itself is deliberately *not* indexed, since
it's already implied by which collection is being queried). An index
creation failure is logged as a warning, not fatal — indexes are an
optimization, not a correctness requirement.

### Batch uploads

`BatchUploader` slices each subject's chunks into `QdrantConfig.batch_size`
pieces (default **100**) before calling `client.upsert()`, logging
`"Batch uploaded: N/M"` per batch — verified with a 5-chunk upload at
`batch_size=2` producing exactly 3 batches. Each batch upsert is wrapped in
the same `retry_with_backoff` helper (`utils/retry.py`) used by the S3
loader and embedding stage, retrying only on transient failures (network
errors, HTTP 429/5xx) — an auth failure (401/403) or a malformed request
fails immediately rather than retrying something that will never succeed.
A single malformed chunk (e.g. a non-UUID `chunk_id`) is skipped with a
logged warning and counted in `UploadResult.skipped_invalid` rather than
aborting the rest of a large batch.

### Scaling strategy

- **Point ID = `chunk_id`**: chunks are minted with a UUID once, in the
  chunking stage, and that same ID is reused as the Qdrant point ID.
  Uploading the same `chunk_id` twice is therefore a Qdrant-native
  update-in-place, not an application-level duplicate check — verified by
  re-uploading a chunk with changed text and confirming the collection's
  point count stayed at 1 while the stored payload reflected the new text.
- **Streaming-friendly batching**: `BatchUploader` only holds one batch's
  worth of points in memory at a time, not the whole upload — appropriate
  for "millions of vectors" without a matching memory footprint.
  `VectorStore` itself holds no per-document state between calls, so it can
  be reused (or instantiated per worker) across an entire textbook corpus
  without any shared mutable state.
- **`delete_document(subject, document_id)`** re-ingests cleanly: deleting
  by a `document_id` filter (indexed) before re-upserting a changed PDF
  avoids orphaned stale chunks if a document's chunk count shrinks between
  runs.
- **`delete_subject(subject)`** drops an entire subject's collection in one
  call — useful for a full re-ingestion of one subject without touching
  any other subject's data.
- **Collection-existence caching**: `CollectionManager` remembers which
  collection names it has already confirmed exist, so `collection_exists`/
  `create_collection` only make a real Qdrant round-trip the *first* time a
  given subject is seen per process — not on every single document's
  upload (or, since `Retriever` shares the same `CollectionManager` via
  `VectorStore`, every single search). Without this, a 100,000-document
  ingestion run would make roughly 100,000 redundant "does this collection
  still exist" calls for collections confirmed on document #1. Verified
  directly in testing: 5 consecutive `upsert_vectors()` calls against the
  same subject trigger exactly 1 underlying `collection_exists` call, not 5.

### Error handling

`vectorstore/exceptions.py` defines `QdrantConnectionError`,
`QdrantAuthenticationError`, `CollectionCreationError`, `BatchUploadError`,
`InvalidPayloadError`, and `QdrantTimeoutError` — all callers need to
handle is this small set, not qdrant-client/httpx internals. HTTP 401/403
responses are mapped to `QdrantAuthenticationError`; network/timeout
failures and retryable 5xx/429 responses are mapped to
`QdrantConnectionError` after retries are exhausted.

### Dependency injection / testability

`VectorStore` accepts an optional `client: QdrantClient` in its
constructor — real Qdrant is never required for tests, since
`qdrant-client` itself ships a fully local, in-memory mode
(`QdrantClient(":memory:")`) that behaves like a real server for
collection/point operations without any external process.

## Retrieval

`retrieval/retriever.py` implements `Retriever`. Its sole job is: **take a
`TeachingContext` (the AI Teacher's current state, not a bare search
string) and return ranked `RetrievedChunk`s.** It does not call an LLM,
orchestrate a tutor turn, handle voice/animation, or expose an API —
`retrieval/` has zero imports from any of those. It composes the two
already-built stages it legitimately depends on: `EmbeddingService` (to
embed the question with the *same* model used to embed the corpus) and
`VectorStore` (to discover collections and reach the Qdrant connection).

```python
retriever = Retriever()   # loads the embedding model once, connects to Qdrant lazily
context = TeachingContext(
    subject="biology", class_name="class11", chapter="Chapter 1: The Cell",
    current_topic="Mitochondria", student_question="Why is it called the powerhouse of the cell?",
)
results = retriever.retrieve(context)   # list[RetrievedChunk], ranked
```

### How retrieval works

This is deliberately **not** a generic "embed the question, search
everything" chatbot flow — every step is narrowed by what the AI Teacher
already knows about the lesson in progress:

1. **`context.subject` selects the collection directly** — if known, only
   that subject's collection is searched (`retrieve_by_subject`); if
   unknown, every collection Qdrant currently has is discovered via
   `VectorStore.list_collections()` and searched, with results merged and
   re-ranked (`retrieve_all`) — never a hardcoded subject list, consistent
   with every other stage in this repository.
2. **The question is embedded**, blended with `current_topic` when present
   (`"Topic: {current_topic}\nQuestion: {student_question}"`) — a short,
   ambiguous student question like *"why does it happen?"* carries almost
   no signal alone, but paired with the topic it does. Verified in testing:
   the same ambiguous question retrieved the correct chunk once
   `current_topic="Photosynthesis"` was attached, where it wouldn't have
   otherwise.
3. **Metadata filters narrow the search** within the selected collection(s)
   (see below).
4. **Qdrant's `query_points` returns the top-K** by cosine similarity,
   with `score_threshold` applied server-side so weak matches never come
   back at all — verified in testing with a deliberately unrelated
   question and a high threshold, returning zero results rather than
   irrelevant ones.
5. **Results are ranked**: deduplicated by `chunk_id` (keeping the
   higher-scoring instance — relevant when `retrieve_all` merges multiple
   collections), sorted by score, truncated to `top_k`.

### Metadata filtering

| Filter | Source | Applied when |
|---|---|---|
| Subject | `TeachingContext.subject` | Selects the collection itself, not a payload filter |
| Class | `TeachingContext.class_name` | Set |
| Chapter | `TeachingContext.chapter` | Set (unset → search the whole subject) |
| Board | `TeachingContext.board` | Set — see note below |
| Language | `TeachingContext.language` | Set — see note below |
| Section | `RetrievalConfig.section` | Set |
| Page range | `RetrievalConfig.page_min` / `page_max` | Either set |

Section and page-range live on `RetrievalConfig`, not `TeachingContext` —
`current_page` (context) is informational teaching state, while a page
*range* filter is a deliberate, opt-in search refinement that shouldn't
apply automatically just because the tutor knows what page a student is
on. All filters are optional and combine with AND (Qdrant's `must`); an
unmatched subject with no filters searches the entire subject.

**Known gap, documented rather than hidden:** the payload written by
`vectorstore/qdrant_client.py` (see that module's docs) does not currently
include `board` or `language` — those fields were never part of the S3
folder structure, parser metadata, or chunk metadata built in earlier
stages. `build_filters` implements board/language filtering faithfully per
this module's spec so it works the moment an ingestion run starts
populating those fields, but until then a `TeachingContext` with `board`
or `language` set will filter against a payload key that doesn't exist —
worth knowing before relying on it, and flagged again in "Suggested
improvements" below.

### Collection selection

Never hardcoded — `retrieve_by_subject`/`retrieve()` call
`VectorStore.collection_exists(subject)` before searching, and
`retrieve_all` calls `VectorStore.list_collections()` fresh on every call.
A subject with no ingested content yet doesn't crash `retrieve()`: it logs
a warning and returns `[]`, since an AI Teacher asking about a not-yet-
ingested subject is a normal, expected outcome, not a bug. (`retrieve_by_subject`
called directly is stricter — it raises `CollectionNotFoundError`, since a
caller invoking it explicitly is asserting that subject should exist.)

### Ranking

`rank_results` is a small, independently testable method: dedupe by
`chunk_id` keeping the best score, sort descending by score, truncate to
`top_k`. It's the single place both the single-collection path and the
multi-collection merge path funnel through, so "no duplicate chunks" holds
regardless of how many collections were searched — verified directly by
merging results across two collections and confirming zero duplicate
`chunk_id`s in the output.

### Output

Every `RetrievedChunk` contains `chunk_id`, `document_id`, `text`, `score`,
`subject`, `class_name`, `chapter`, `section`, `page_number`, `source_pdf`,
and `metadata` (the full original payload's nested metadata dict) — built
directly from the exact payload shape `vectorstore/qdrant_client.py`
writes, so no field is guessed or reverse-engineered.

### Error handling

`retrieval/exceptions.py` defines `CollectionNotFoundError`,
`QueryEmbeddingError`, `SearchError`, `SearchTimeoutError`, and
`RetrievalValidationError`. A search's retries (`utils/retry.py`, same
helper as every other stage) are bounded by an overall `timeout_seconds`
wall-clock ceiling — retries happen *inside* that window rather than
extending it, appropriate for a low-latency, tutor-facing search.

### Dependency injection / testability

`Retriever` accepts `vector_store`/`embedding_service` in its constructor,
so tests never need a real Qdrant server or the real (multi-GB)
`BAAI/bge-m3` model — verified in testing using `qdrant-client`'s in-memory
mode plus a small real embedding model. `RetrievalConfig` is fixed at
construction (matching every other stage's pattern in this repo) rather
than accepted per-call; since `Retriever` itself is a thin, cheap-to-build
facade around already-loaded collaborators, the recommended way to vary
`top_k`/filters per query is constructing a second `Retriever` that shares
the same `vector_store`/`embedding_service` instances, not adding a
parallel per-call config API.

## Pipeline Orchestrator

`pipeline/ingest_pipeline.py` implements `IngestionPipeline`. Its sole job
is: **coordinate `S3Loader → DoclingParser → LlamaChunker →
EmbeddingService → VectorStore` for every discovered PDF, one document at a
time.** All business logic stays inside each stage — the orchestrator only
sequences calls, tracks state, and decides what to retry/skip/continue.
None of the five modules know the orchestrator exists; each is still fully
usable and testable standalone, exactly as before.

```python
pipeline = IngestionPipeline()          # wires up all five stages with their own defaults
stats = pipeline.run()                  # entire bucket
stats = pipeline.run_subject("biology") # one subject
state = pipeline.run_document(pdf)      # one PDF (an S3PdfObject from S3Loader.discover_pdf_files)
```

### Pipeline execution — the 7-step workflow

For every discovered PDF, `run_document` runs:

1. **Check if already processed** — look up the PDF's S3 key in the
   checkpoint; if it previously succeeded and `overwrite_existing` is
   `False` (default), skip it entirely (no download, no re-parse, nothing).
2. **Download** via `S3Loader.download_file`.
3. **Parse** via `DoclingParser.parse`.
4. **Chunk** via `LlamaChunker.chunk`.
5. **Generate embeddings** via `EmbeddingService.embed_chunks`.
6. **Store vectors** via `VectorStore.upsert_vectors`.
7. **Mark completed** — update and persist the checkpoint.

A failure at any step is caught *inside* `run_document` — it never
propagates to the caller — so `run()`/`run_subject()` always move on to the
next PDF regardless of what happened to the current one. Verified directly
in testing: a PDF that fails at the parsing step every time still lets
every other PDF in the batch complete successfully.

### Checkpointing

Every document's outcome is recorded in a `PipelineState` — a Pydantic
model persisted to `PipelineConfig.checkpoint_path` (default
`pipeline_state.json`) as compact JSON, written atomically (temp file +
replace, so a crash mid-write never corrupts the checkpoint), by default
**after every single document**. The checkpoint key is the document's **S3
key** (`biology/class11/chapter1.pdf`), not its parser-assigned
`document_id` — the S3 key is known the moment a PDF is discovered, before
it's ever downloaded or parsed, so "already processed" can be checked
before any expensive work happens; `document_id` is filled in once parsing
succeeds, for cross-referencing with `VectorStore.delete_document`.

`PipelineState.save()` rewrites the *entire* checkpoint file each time it's
called — fine at hundreds or a few thousand documents, but the cost grows
with how many documents have completed so far, and at "100,000 PDFs" scale
saving after literally every document means the cumulative bytes written
over a full run reach into the terabytes. `PipelineConfig.checkpoint_save_interval`
(default `1`, preserving the strongest crash-safety guarantee) lets an
operator trade a bounded amount of that guarantee — at most
`checkpoint_save_interval - 1` documents' *disk-persisted* progress can be
lost on a hard crash, though nothing is corrupted and re-processing a
document is always safe (every downstream write is idempotent by
`chunk_id`) — for meaningfully less I/O on very large runs. In-memory state
is never throttled by this setting: `error_report()` and the returned
`PipelineStatistics` always reflect every document processed in the current
run regardless of the save interval, and a final flush always happens at
the end of `run()`/`run_subject()` (including on `Ctrl+C` or an unexpected
exception during discovery), so a run never *silently* ends with
unpersisted progress — verified directly in testing with
`checkpoint_save_interval=10` across 4 documents: exactly one disk write
occurred (the final flush), not four, and every document's success was
still correctly persisted.

### Resume mechanism

Resuming is not a special mode — it's just what happens naturally on the
next `run()`/`run_subject()` call, in a brand-new process, because
`IngestionPipeline.__init__` loads `PipelineState` from
`checkpoint_path` if it exists. Verified directly in testing: run a batch
to completion, construct a **completely new** `IngestionPipeline` instance
(simulating a fresh process after a crash/restart) pointed at the same
checkpoint file, call `run()` again — every previously-completed document
is skipped, with the underlying parser never invoked a second time for any
of them. "Resume from the last completed document" falls out for free from
"never reprocess anything already marked successful."

### Failure recovery

Each of the 7 steps is retried independently (`utils/retry.py`'s shared
backoff helper, the same one every other stage already uses) up to
`PipelineConfig.max_retries` times before the document is marked failed —
deliberately a **uniform retry-then-skip policy**, not fine-grained
transient/permanent error classification: each underlying stage (S3Loader,
EmbeddingService, VectorStore) already retries *its own* transient failures
internally before ever raising up to the orchestrator, so by the time an
exception reaches here, further retries are a coarser "try the whole step
again in a moment" safety net rather than a first line of defense. A
corrupted or password-protected PDF still costs `max_retries` attempts
before being skipped (see "Suggested improvements" in the review below for
a cheaper alternative). Retrying only re-runs the *step that failed* — a
document that fails at chunking doesn't get re-downloaded or re-parsed on
retry, since download/parse already succeeded and their results are held
in memory for the rest of that `run_document` call.

`DocumentState` (per document, inside `PipelineState`) tracks exactly what
the spec asks for: `s3_key`/`document_id`, `stage` (which of the 7 steps
it's in or last failed at), `success`, `last_error`, `retry_count`,
`first_attempted_at`/`last_attempted_at`/`completed_at`, and
`processing_time_seconds`.

### Batch processing

| Scope | Method |
|---|---|
| Single PDF | `run_document(pdf: S3PdfObject)` |
| Entire subject | `run_subject(subject: str)` |
| Entire bucket | `run()` |

All three share the same per-document logic and checkpoint, so mixing
scopes across runs (e.g. `run_subject("biology")` today,
`run()` tomorrow) is safe — biology's already-completed documents are
still skipped when the full-bucket run reaches them.
`PipelineConfig.parallel_workers` is accepted and stored today as a
forward-looking, currently no-op field — see "Suggested improvements."

### Progress reporting & metrics

Progress is reported as structured log lines (`Progress: subject=...
document=... stage=... completed=N/M remaining=K elapsed=Xs eta=Ys`) rather
than a live-redrawing terminal bar — deliberately, since this pipeline is
meant to run unattended over thousands of PDFs (a container, cron job, or
CI log), where a live progress widget's control codes get mangled by log
aggregation; plain sequential lines with elapsed/ETA hold up better there
than a nicer-looking live bar would. `PipelineStatistics` (returned by
`run`/`run_subject`, reset at the start of each call) collects
`total_pdfs`, `successful_pdfs`, `failed_pdfs`, `skipped_pdfs`,
`total_chunks`, `total_embeddings`, `total_stored_vectors`, and
`average_processing_time_seconds` — verified directly in testing to add up
correctly across a mixed batch of successes and one induced failure.
`error_report()` returns every currently-failed `DocumentState`, and the
final `"Pipeline finished"` log line is followed by one line per failure
if any occurred.

### Configuration

| Field (`PipelineConfig`) | Default | Meaning |
|---|---|---|
| `overwrite_existing` | `False` | Force reprocessing of already-completed documents |
| `max_retries` | `3` | Retry attempts per step before marking a document failed |
| `batch_size` | `100` | Reserved for future batch-level tuning passed to sub-stage configs |
| `parallel_workers` | `1` | Reserved for future parallel execution (currently sequential only) |
| `dry_run` | `False` | Run discovery + checkpoint checks only; no download/parse/embed/store |
| `download_root` | `downloads/` | Passed through to the default `S3Loader` |
| `checkpoint_path` | `pipeline_state.json` | Where `PipelineState` is persisted |
| `checkpoint_save_interval` | `1` | Documents between disk saves (see "Checkpointing" above); always flushed at the end of a run regardless |
| `show_progress` | `True` | Emit the per-document progress log line |

### Error handling

Per-document failures never raise out of `run_document`/`run`/`run_subject`
— they're recorded in `PipelineState` and surfaced via the final log
summary and `error_report()`. `pipeline/exceptions.py` defines
`PipelineError` and `CheckpointError`, reserved for orchestrator-level
problems unrelated to any single document (a corrupted/unwritable
checkpoint file).

## Pipeline Status

Every stage described above is implemented:

1. ~~`S3Loader` — list and download PDFs from S3 into local/temp storage.~~ **Implemented.**
2. ~~`DoclingParser` — parse each PDF into a structured, layout-aware document.~~ **Implemented.**
3. ~~`LlamaChunker` — split structured documents into retrieval-sized chunks.~~ **Implemented.**
4. ~~`EmbeddingService` — generate dense embeddings for each chunk via `BAAI/bge-m3`.~~ **Implemented.**
5. ~~`VectorStore` — upsert embeddings and metadata into Qdrant.~~ **Implemented.**
6. ~~`Retriever` — search Qdrant for chunks relevant to a teaching moment.~~ **Implemented.**
7. ~~`IngestionPipeline` — orchestrate the ingestion stages above, resumably.~~ **Implemented.**

Once implemented, the pipeline should be resumable, idempotent (safe to
re-run without duplicating vectors), and horizontally scalable across
workers processing independent S3 objects.
