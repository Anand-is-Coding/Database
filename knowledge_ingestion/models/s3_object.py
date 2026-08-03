"""Domain model for a PDF object discovered in S3.

Shared between discovery and download methods on `S3Loader` so the two
sides of that contract stay in sync without passing around raw boto3 dicts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class S3PdfObject(BaseModel):
    """A single PDF discovered under `<subject>/<class_name>/...` in S3.

    `subject` and `class_name` are derived purely from the object key's
    folder structure — never hardcoded — so newly uploaded subjects or
    classes are represented automatically.
    """

    subject: str
    class_name: str
    key: str
    filename: str
    size_bytes: int
    last_modified: datetime
    etag: str
