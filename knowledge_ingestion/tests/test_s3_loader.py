"""Unit tests for S3Loader — dependency-injected mock boto3 client, no network."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from loaders.exceptions import BucketNotFoundError, ObjectNotFoundError, S3AccessDeniedError
from loaders.s3_loader import S3Loader


def _paginator(pages: list[dict]) -> MagicMock:
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


def test_discover_subjects_reflects_bucket_contents_with_no_hardcoding():
    client = MagicMock()
    client.get_paginator.return_value = _paginator(
        [{"CommonPrefixes": [{"Prefix": "biology/"}, {"Prefix": "physics/"}]}]
    )
    loader = S3Loader(bucket_name="test-bucket", s3_client=client, max_retries=1)

    assert loader.discover_subjects() == ["biology", "physics"]

    # A brand-new subject appearing tomorrow requires zero code changes.
    client.get_paginator.return_value = _paginator(
        [{"CommonPrefixes": [{"Prefix": "computer_science/"}]}]
    )
    loader.clear_cache()
    assert loader.discover_subjects() == ["computer_science"]


def test_discover_pdf_files_filters_non_pdfs_and_derives_class_name():
    client = MagicMock()
    now = datetime.now(timezone.utc)
    client.get_paginator.return_value = _paginator(
        [
            {
                "Contents": [
                    {"Key": "biology/class11/chapter1.pdf", "Size": 100, "LastModified": now, "ETag": '"abc"'},
                    {"Key": "biology/class11/", "Size": 0, "LastModified": now, "ETag": '"x"'},
                    {"Key": "biology/notes.txt", "Size": 5, "LastModified": now, "ETag": '"y"'},
                ]
            }
        ]
    )
    loader = S3Loader(bucket_name="test-bucket", s3_client=client, max_retries=1)
    pdfs = loader.discover_pdf_files("biology")

    assert len(pdfs) == 1
    assert pdfs[0].class_name == "class11"
    assert pdfs[0].filename == "chapter1.pdf"


def test_verify_connection_maps_404_to_bucket_not_found():
    client = MagicMock()
    client.head_bucket.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket")
    loader = S3Loader(bucket_name="missing-bucket", s3_client=client, max_retries=1)

    with pytest.raises(BucketNotFoundError):
        loader.verify_connection()


def test_verify_connection_maps_403_to_access_denied():
    client = MagicMock()
    client.head_bucket.side_effect = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket")
    loader = S3Loader(bucket_name="secret-bucket", s3_client=client, max_retries=1)

    with pytest.raises(S3AccessDeniedError):
        loader.verify_connection()


def test_download_file_skips_existing_and_overwrite_forces_redownload(tmp_path):
    from models.s3_object import S3PdfObject

    client = MagicMock()
    loader = S3Loader(bucket_name="test-bucket", s3_client=client, download_root=tmp_path, max_retries=1)
    pdf = S3PdfObject(
        subject="biology",
        class_name="class11",
        key="biology/class11/chapter1.pdf",
        filename="chapter1.pdf",
        size_bytes=10,
        last_modified=datetime.now(timezone.utc),
        etag="abc",
    )

    def fake_download_file(bucket, key, dest):
        from pathlib import Path

        Path(dest).write_bytes(b"%PDF-1.4 fake")

    client.download_file.side_effect = fake_download_file

    path1 = loader.download_file(pdf)
    assert path1.exists()
    assert client.download_file.call_count == 1

    # Second call skips — no new S3 call, no crash even though the
    # destination file already exists on Windows (Path.replace(), not
    # Path.rename(), matters for the overwrite=True case below).
    path2 = loader.download_file(pdf)
    assert client.download_file.call_count == 1
    assert path2 == path1

    loader.download_file(pdf, overwrite=True)
    assert client.download_file.call_count == 2
    assert path1 == tmp_path / "biology" / "class11" / "chapter1.pdf"


def test_download_file_maps_404_to_object_not_found(tmp_path):
    from models.s3_object import S3PdfObject

    client = MagicMock()
    client.download_file.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
    loader = S3Loader(bucket_name="test-bucket", s3_client=client, download_root=tmp_path, max_retries=1)
    pdf = S3PdfObject(
        subject="physics",
        class_name="class12",
        key="physics/class12/missing.pdf",
        filename="missing.pdf",
        size_bytes=10,
        last_modified=datetime.now(timezone.utc),
        etag="abc",
    )

    with pytest.raises(ObjectNotFoundError):
        loader.download_file(pdf)
