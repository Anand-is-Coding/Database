"""Shared Qdrant error classification.

Both `vectorstore/qdrant_client.py` (writes) and `retrieval/retriever.py`
(reads) talk to the same Qdrant client and need to decide the same thing —
"is this exception worth retrying?" — so that logic lives here once instead
of being copy-pasted between the two modules.
"""

from __future__ import annotations

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def is_transient_qdrant_error(exc: Exception) -> bool:
    """Decide whether an exception raised by qdrant-client is worth retrying.

    `ResponseHandlingException` wraps network/timeout failures (always
    worth retrying); `UnexpectedResponse` wraps HTTP error responses, only
    worth retrying for the status codes above (429/5xx) — not 4xx client
    errors like bad requests or auth failures, which won't succeed on retry.
    """
    if isinstance(exc, ResponseHandlingException):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code in RETRYABLE_HTTP_STATUSES
    return False
