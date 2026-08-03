"""S3 loading module — responsible for retrieving source PDFs from AWS S3.

Single responsibility: discover and download PDFs from S3, preserving the
bucket's folder hierarchy locally. Parsing, chunking, embedding, and vector
storage are handled by other modules.

Subject and class names are never hardcoded — they are discovered from the
bucket's own folder structure (`discover_subjects`, `discover_classes`), so
new subjects/classes uploaded to S3 are picked up automatically.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ReadTimeoutError,
)
from tqdm import tqdm

from config.logging import get_logger
from config.settings import settings
from loaders.exceptions import (
    BucketNotFoundError,
    ObjectNotFoundError,
    S3AccessDeniedError,
    S3AuthenticationError,
    S3ConnectionError,
    S3DownloadError,
    S3LoaderError,
)
from models.s3_object import S3PdfObject
from utils.retry import retry_with_backoff

logger = get_logger(__name__)

# Our own retry_with_backoff is the single, visible retry mechanism used by
# this loader, so botocore's built-in retrying is turned down to one attempt
# to avoid two opaque retry layers stacking on top of each other.
_BOTO_CONFIG = BotoConfig(
    connect_timeout=10,
    read_timeout=60,
    retries={"max_attempts": 1, "mode": "standard"},
)

_TRANSIENT_CLIENT_ERROR_CODES = {
    "500",
    "502",
    "503",
    "504",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
    "Throttling",
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
}


def _is_transient_error(exc: Exception) -> bool:
    """Decide whether an exception raised by boto3 is worth retrying."""
    if isinstance(exc, (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _TRANSIENT_CLIENT_ERROR_CODES
    return False


class S3Loader:
    """Discovers and downloads PDFs from the configured S3 bucket.

    All AWS access goes through an injected boto3 client, so tests can pass
    a mock/stub client instead of talking to real S3. The loader holds no
    module-level/global state — all caching lives on the instance.
    """

    def __init__(
        self,
        bucket_name: str | None = None,
        s3_client: Any | None = None,
        download_root: str | Path = "downloads",
        max_retries: int = 3,
    ) -> None:
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.download_root = Path(download_root)
        self.max_retries = max_retries
        self._client = s3_client or self._build_default_client()

        self._subjects_cache: list[str] | None = None
        self._classes_cache: dict[str, list[str]] = {}
        self._pdf_cache: dict[str, list[S3PdfObject]] = {}

    @staticmethod
    def _build_default_client() -> Any:
        """Build a boto3 S3 client from settings.

        Explicit keys are only passed when both are configured; otherwise
        boto3's default credential chain (env vars, shared config, IAM
        instance/task role) is left to resolve credentials, which is the
        production-appropriate default.
        """
        secret_key = settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
        client_kwargs: dict[str, Any] = {"region_name": settings.AWS_REGION, "config": _BOTO_CONFIG}
        if settings.AWS_ACCESS_KEY_ID and secret_key:
            client_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = secret_key
        return boto3.client("s3", **client_kwargs)

    def clear_cache(self) -> None:
        """Drop cached discovery results, forcing the next call to hit S3 again."""
        self._subjects_cache = None
        self._classes_cache.clear()
        self._pdf_cache.clear()

    # ------------------------------------------------------------------
    # Connection / credential verification
    # ------------------------------------------------------------------

    def verify_connection(self) -> bool:
        """Verify the bucket exists and is reachable with the configured credentials.

        A single `head_bucket` call verifies credentials, network
        reachability, and bucket access together — no need for separate
        API calls to check each concern independently.
        """

        def _head_bucket() -> None:
            self._client.head_bucket(Bucket=self.bucket_name)

        try:
            retry_with_backoff(
                _head_bucket, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise S3AuthenticationError(f"AWS credentials are missing or invalid: {exc}") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("403", "AccessDenied"):
                raise S3AccessDeniedError(
                    f"Access denied to bucket '{self.bucket_name}'. Check IAM permissions."
                ) from exc
            if code in ("404", "NoSuchBucket"):
                raise BucketNotFoundError(f"Bucket '{self.bucket_name}' does not exist.") from exc
            raise S3ConnectionError(
                f"Failed to connect to bucket '{self.bucket_name}': {exc}"
            ) from exc
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as exc:
            raise S3ConnectionError(f"Network failure while reaching S3: {exc}") from exc

        logger.info("Connection established: bucket '%s' is reachable.", self.bucket_name)
        return True

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_subjects(self, *, use_cache: bool = True) -> list[str]:
        """Discover every subject by listing first-level "folders" in the bucket.

        Subjects are never hardcoded — whatever prefixes exist in S3 today
        (biology, physics, ...) or are added tomorrow (history, geography,
        ...) are returned automatically.
        """
        if use_cache and self._subjects_cache is not None:
            return self._subjects_cache

        def _list_subjects() -> list[str]:
            found: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Delimiter="/"):
                for common_prefix in page.get("CommonPrefixes", []):
                    found.append(common_prefix["Prefix"].rstrip("/"))
            return sorted(found)

        subjects = self._call_with_retry(_list_subjects, action=f"list subjects in bucket '{self.bucket_name}'")
        logger.info("Discovered %d subject(s): %s", len(subjects), ", ".join(subjects) or "none")
        self._subjects_cache = subjects
        return subjects

    def discover_classes(self, subject: str, *, use_cache: bool = True) -> list[str]:
        """Discover every class (second-level folder) under a subject."""
        if use_cache and subject in self._classes_cache:
            return self._classes_cache[subject]

        prefix = f"{subject}/"

        def _list_classes() -> list[str]:
            found: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"):
                for common_prefix in page.get("CommonPrefixes", []):
                    found.append(common_prefix["Prefix"][len(prefix) :].rstrip("/"))
            return sorted(found)

        classes = self._call_with_retry(_list_classes, action=f"list classes for subject '{subject}'")
        if not classes:
            logger.warning("No classes found for subject '%s'.", subject)
        else:
            logger.info("Discovered %d class(es) for '%s': %s", len(classes), subject, ", ".join(classes))
        self._classes_cache[subject] = classes
        return classes

    def discover_pdf_files(self, subject: str, *, use_cache: bool = True) -> list[S3PdfObject]:
        """Discover every PDF nested (at any depth) under a subject."""
        if use_cache and subject in self._pdf_cache:
            return self._pdf_cache[subject]

        prefix = f"{subject}/"

        def _list_pdfs() -> list[S3PdfObject]:
            found: list[S3PdfObject] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/") or not key.lower().endswith(".pdf"):
                        continue
                    found.append(self._to_pdf_object(subject, obj))
            return found

        pdfs = self._call_with_retry(_list_pdfs, action=f"list PDFs for subject '{subject}'")
        if not pdfs:
            logger.warning("No PDF files found under subject '%s'.", subject)
        else:
            logger.info("Found %d PDF file(s) under subject '%s'.", len(pdfs), subject)
        self._pdf_cache[subject] = pdfs
        return pdfs

    def list_all_pdfs(self, *, use_cache: bool = True) -> list[S3PdfObject]:
        """Discover every PDF across every subject in the bucket."""
        all_pdfs: list[S3PdfObject] = []
        for subject in self.discover_subjects(use_cache=use_cache):
            all_pdfs.extend(self.discover_pdf_files(subject, use_cache=use_cache))
        logger.info("Total PDFs discovered across all subjects: %d", len(all_pdfs))
        return all_pdfs

    def _to_pdf_object(self, subject: str, obj: dict[str, Any]) -> S3PdfObject:
        key = obj["Key"]
        relative_parts = key[len(subject) + 1 :].split("/")
        filename = relative_parts[-1]
        class_name = relative_parts[0] if len(relative_parts) > 1 else ""
        return S3PdfObject(
            subject=subject,
            class_name=class_name,
            key=key,
            filename=filename,
            size_bytes=obj["Size"],
            last_modified=obj["LastModified"],
            etag=obj.get("ETag", "").strip('"'),
        )

    # ------------------------------------------------------------------
    # Metadata / existence checks
    # ------------------------------------------------------------------

    def file_exists(self, key: str) -> bool:
        """Check whether an object key exists in the bucket."""
        try:
            self.get_object_metadata(key)
            return True
        except ObjectNotFoundError:
            return False

    def get_object_metadata(self, key: str) -> dict[str, Any]:
        """Fetch metadata for a single object without downloading it."""

        def _head_object() -> dict[str, Any]:
            return self._client.head_object(Bucket=self.bucket_name, Key=key)

        try:
            response = retry_with_backoff(
                _head_object, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise ObjectNotFoundError(f"Object not found in S3: {key}") from exc
            raise S3ConnectionError(f"Failed to fetch metadata for '{key}': {exc}") from exc
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as exc:
            raise S3ConnectionError(f"Network failure while fetching metadata for '{key}': {exc}") from exc

        return {
            "size_bytes": response["ContentLength"],
            "last_modified": response["LastModified"],
            "etag": response.get("ETag", "").strip('"'),
            "content_type": response.get("ContentType"),
        }

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def download_file(self, pdf: S3PdfObject, *, overwrite: bool = False) -> Path:
        """Download a single PDF, preserving its `subject/class/...` path locally.

        Skips the download entirely (no S3 call) if the destination already
        exists locally and `overwrite` is False.
        """
        destination = self.download_root / pdf.key
        if destination.exists() and not overwrite:
            logger.info("Skipping (already exists): %s", pdf.key)
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.parent / f"{destination.name}.part"

        def _do_download() -> None:
            self._client.download_file(self.bucket_name, pdf.key, str(tmp_destination))

        try:
            retry_with_backoff(
                _do_download, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except ClientError as exc:
            tmp_destination.unlink(missing_ok=True)
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                raise ObjectNotFoundError(f"Object not found in S3: {pdf.key}") from exc
            raise S3DownloadError(f"Failed to download '{pdf.key}': {exc}") from exc
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as exc:
            tmp_destination.unlink(missing_ok=True)
            raise S3ConnectionError(f"Network failure while downloading '{pdf.key}': {exc}") from exc

        # `.replace()` (not `.rename()`) so this overwrites atomically on
        # Windows too — `Path.rename()` raises FileExistsError there if the
        # destination already exists, unlike POSIX rename semantics.
        tmp_destination.replace(destination)
        logger.info("Downloaded: %s -> %s", pdf.key, destination)
        return destination

    def download_subject(
        self, subject: str, *, overwrite: bool = False, max_workers: int = 8
    ) -> list[Path]:
        """Download every PDF discovered under a single subject."""
        pdfs = self.discover_pdf_files(subject)
        if not pdfs:
            return []
        return self._download_many(pdfs, overwrite=overwrite, max_workers=max_workers, desc=subject)

    def download_all(
        self, *, overwrite: bool = False, max_workers: int = 8
    ) -> dict[str, list[Path]]:
        """Download every PDF across every discovered subject."""
        subjects = self.discover_subjects()
        results: dict[str, list[Path]] = {}
        for subject in subjects:
            results[subject] = self.download_subject(
                subject, overwrite=overwrite, max_workers=max_workers
            )
        total = sum(len(paths) for paths in results.values())
        logger.info("Download complete: %d file(s) across %d subject(s).", total, len(subjects))
        return results

    def _download_many(
        self, pdfs: list[S3PdfObject], *, overwrite: bool, max_workers: int, desc: str
    ) -> list[Path]:
        results: list[Path] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.download_file, pdf, overwrite=overwrite): pdf for pdf in pdfs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {desc}"):
                pdf = futures[future]
                try:
                    results.append(future.result())
                except S3LoaderError as exc:
                    logger.error("Failed to download '%s': %s", pdf.key, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, func: Any, *, action: str) -> Any:
        try:
            return retry_with_backoff(
                func, should_retry=_is_transient_error, max_attempts=self.max_retries
            )
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise S3AuthenticationError(f"AWS credentials are missing or invalid: {exc}") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("403", "AccessDenied"):
                raise S3AccessDeniedError(f"Access denied while trying to {action}.") from exc
            if code in ("404", "NoSuchBucket"):
                raise BucketNotFoundError(f"Bucket '{self.bucket_name}' does not exist.") from exc
            raise S3ConnectionError(f"Failed to {action}: {exc}") from exc
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as exc:
            raise S3ConnectionError(f"Network failure while trying to {action}: {exc}") from exc
