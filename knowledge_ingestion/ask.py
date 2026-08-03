"""Interactive CLI for asking questions against the ingested knowledge base.

Composition root only: wires `Retriever` and `AnswerGenerator` together and
runs a question/answer loop in the terminal. No retrieval or generation
logic lives here.
"""

from __future__ import annotations

import os

# Same rationale as main.py: keep native threading backends single-threaded
# so loading the BGE-M3 model to embed a query doesn't add memory pressure
# on top of whatever else is running.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from config.logging import get_logger
from generation.exceptions import GenerationError
from generation.gemini_generator import AnswerGenerator
from models.retrieval import TeachingContext
from retrieval.exceptions import RetrievalError
from retrieval.retriever import Retriever
from vectorstore.qdrant_client import VectorStore

logger = get_logger(__name__)


def resolve_subject_choice(choice: str, known_subjects: list[str]) -> str | None:
    """Map a numeric menu choice to a subject name.

    '0' (or blank) means "search all collections". Raises `ValueError` with
    a user-facing message for anything else invalid - scoping to one
    collection (instead of `retrieve_all` fanning out to every collection)
    is the single biggest lever on retrieval latency, so this exists to
    make picking a subject a typo-proof number instead of typing the
    (case-sensitive, sometimes odd, e.g. "Physics11,12") collection name.
    """
    if not choice or choice == "0":
        return None
    if not choice.isdigit():
        raise ValueError(f"'{choice}' is not a valid choice.")
    index = int(choice)
    if not 1 <= index <= len(known_subjects):
        raise ValueError(f"{index} is out of range.")
    return known_subjects[index - 1]


def ask(retriever: Retriever, generator: AnswerGenerator, question: str, subject: str | None = None) -> None:
    context = TeachingContext(student_question=question, subject=subject)
    chunks = retriever.retrieve(context)
    if not chunks:
        print("\nNo relevant content found in the knowledge base for that question.\n")
        return

    answer = generator.generate(question, chunks)
    print(f"\n{answer.text}\n")
    print("Sources:")
    for chunk in answer.sources:
        location = " / ".join(
            part
            for part in (chunk.subject, chunk.chapter, f"page {chunk.page_number}" if chunk.page_number else None)
            if part
        )
        print(f"  - {location or chunk.source_pdf or chunk.document_id} (score {chunk.score:.3f})")
    print()


def print_subject_menu(known_subjects: list[str]) -> None:
    print("Subjects:")
    print("  0. All subjects")
    for i, name in enumerate(known_subjects, start=1):
        print(f"  {i}. {name}")


def main() -> None:
    retriever = Retriever()
    generator = AnswerGenerator()
    known_subjects = sorted(VectorStore().list_collections())

    print("YoTutor Q&A -- type 'quit' at either prompt to exit.\n")

    while True:
        print_subject_menu(known_subjects)
        try:
            choice = input("\nPick a subject number: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice.lower() in {"quit", "exit"}:
            break

        try:
            subject = resolve_subject_choice(choice, known_subjects)
        except ValueError as exc:
            print(f"{exc} Try again.\n")
            continue

        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        try:
            ask(retriever, generator, question, subject)
        except (RetrievalError, GenerationError) as exc:
            logger.error("Failed to answer question: %s", exc)
            print(f"\nSomething went wrong: {exc}\n")


if __name__ == "__main__":
    main()
