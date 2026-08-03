"""Custom exceptions for the S3 loading stage.

Kept local to `loaders/` since these errors are specific to this stage's
concerns (connectivity, credentials, object lookup) and should not leak
implementation details (boto3/botocore types) to callers.
"""

from __future__ import annotations


class S3LoaderError(Exception):
    """Base exception for all S3 loader failures."""


class S3AuthenticationError(S3LoaderError):
    """AWS credentials are missing, malformed, or rejected."""


class S3AccessDeniedError(S3LoaderError):
    """Credentials are valid but lack permission for the requested action."""


class BucketNotFoundError(S3LoaderError):
    """The configured bucket does not exist."""


class S3ConnectionError(S3LoaderError):
    """The S3 endpoint could not be reached (network failure, timeout)."""


class ObjectNotFoundError(S3LoaderError):
    """A requested S3 object key does not exist."""


class S3DownloadError(S3LoaderError):
    """A file download failed after all retry attempts were exhausted."""
