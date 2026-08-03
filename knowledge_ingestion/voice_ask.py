"""Interactive voice CLI for asking questions against the ingested
knowledge base.

Composition root only: wires `Retriever`, `AnswerGenerator`,
`SpeechToTextService`, and `TextToSpeechService` together and runs a voice
question/answer loop in the terminal. No retrieval, generation, or voice
logic lives here.
"""

from __future__ import annotations

import os

# Same rationale as main.py/ask.py: keep native threading backends
# single-threaded so loading the BGE-M3 model doesn't add memory pressure
# on top of whatever else is running.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ask import print_subject_menu, resolve_subject_choice
from config.logging import get_logger
from generation.exceptions import GenerationError
from generation.gemini_generator import AnswerGenerator
from models.retrieval import TeachingContext
from retrieval.exceptions import RetrievalError
from retrieval.retriever import Retriever
from vectorstore.qdrant_client import VectorStore
from voice.exceptions import VoiceError
from voice.microphone import record_until_enter
from voice.speech_to_text import SpeechToTextService
from voice.text_to_speech import TextToSpeechService

logger = get_logger(__name__)


def ask_by_voice(
    retriever: Retriever,
    generator: AnswerGenerator,
    stt: SpeechToTextService,
    tts: TextToSpeechService,
    subject: str | None,
) -> None:
    input("Press Enter, then ask your question. Press Enter again to stop recording.")
    print("Recording... press Enter to stop.")
    audio_bytes = record_until_enter()

    print("Transcribing...")
    question = stt.transcribe(audio_bytes)
    print(f"You asked: {question}")

    context = TeachingContext(student_question=question, subject=subject)
    chunks = retriever.retrieve(context)
    if not chunks:
        message = "I couldn't find anything relevant to that question in the knowledge base."
        print(message)
        tts.speak(message, subject)
        return

    answer = generator.generate(question, chunks)
    print(f"\n{answer.text}\n")
    tts.speak(answer.text, subject)


def main() -> None:
    retriever = Retriever()
    generator = AnswerGenerator()
    stt = SpeechToTextService()
    tts = TextToSpeechService()
    known_subjects = sorted(VectorStore().list_collections())

    print("YoTutor Voice Q&A -- type 'quit' at the subject prompt to exit.\n")

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
            ask_by_voice(retriever, generator, stt, tts, subject)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except (RetrievalError, GenerationError, VoiceError) as exc:
            logger.error("Failed to answer question by voice: %s", exc)
            print(f"\nSomething went wrong: {exc}\n")
        print()


if __name__ == "__main__":
    main()
