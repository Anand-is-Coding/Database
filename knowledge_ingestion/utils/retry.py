"""Generic retry helper for transient failures (network blips, throttling).

Deliberately framework-agnostic: it knows nothing about S3 or boto3, so it
can be reused by future stages (parsing, embedding, Qdrant) without
duplicating backoff logic.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
) -> T:
    """Call `func`, retrying with exponential backoff + jitter on transient errors.

    `should_retry` decides whether a caught exception is worth retrying; a
    non-retryable exception (or the final attempt) propagates immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
            logger.warning(
                "Transient error on attempt %d/%d: %s — retrying in %.2fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
