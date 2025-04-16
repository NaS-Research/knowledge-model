"""
upload_s3.py
------------
Reusable helpers for uploading artifacts to AWS S3.

• PDFs             → nas-knowledge-model-pdfs/<filename>
• train.jsonl      → nas-knowledge-model-dataset/train.jsonl
• clean chunks     → nas-knowledge-model-dataset/clean/YYYY/MM/<filename>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import boto3

S3_PDF_BUCKET = os.getenv("S3_PDF_BUCKET", "nas-knowledge-model-pdfs")
S3_DATASET_BUCKET = os.getenv("S3_DATASET_BUCKET", "nas-knowledge-model-dataset")

s3_client = boto3.client("s3")


def _upload_file(
    file_path: Union[str, Path],
    bucket: str,
    key: str,
    extra_args: Optional[dict] = None,
) -> str:
    """Upload a single local file and return its HTTPS S3 URL."""
    path = Path(file_path)
    s3_client.upload_file(str(path), bucket, key, ExtraArgs=extra_args or {})
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def upload_directory(
    dir_path: Union[str, Path],
    bucket: str,
    prefix: str = "",
    recurse: bool = True,
) -> list[str]:
    """
    Upload an entire directory to S3 recursively.

    Args:
        dir_path: Path to the root directory.
        bucket: S3 bucket name.
        prefix: Prefix to prepend to object keys.
        recurse: Whether to scan subdirectories.

    Returns:
        List of public HTTPS URLs of uploaded files.
    """
    base = Path(dir_path)
    urls: list[str] = []

    for path in base.rglob("*" if recurse else "*.*"):
        if path.is_file():
            rel = path.relative_to(base)
            key = f"{prefix}/{rel.as_posix()}".lstrip("/")
            urls.append(_upload_file(path, bucket, key))

    return urls


def upload_pdf_to_s3(file_path: str) -> str:
    """
    Upload a single PDF to the dedicated PDF bucket.

    Args:
        file_path: Path to the PDF file.

    Returns:
        HTTPS URL of uploaded PDF.
    """
    filename = os.path.basename(file_path)
    return _upload_file(file_path, S3_PDF_BUCKET, filename)


def upload_dataset_to_s3(file_path: str) -> str:
    """
    Upload a dataset file (e.g. train.jsonl) to the dataset root.

    Args:
        file_path: Path to the dataset.

    Returns:
        HTTPS URL of uploaded dataset.
    """
    filename = os.path.basename(file_path)
    return _upload_file(file_path, S3_DATASET_BUCKET, filename)


def upload_clean_chunk(file_path: str, year: str, month: str) -> str:
    """
    Upload a single chunk file to clean/YYYY/MM/ in the dataset bucket.

    Args:
        file_path: Path to the JSONL chunk.
        year: 4-digit year.
        month: 2-digit month.

    Returns:
        HTTPS URL of uploaded file.
    """
    filename = os.path.basename(file_path)
    key = f"clean/{year}/{month}/{filename}"
    return _upload_file(file_path, S3_DATASET_BUCKET, key)


def upload_clean_tree(dir_path: str) -> list[str]:
    """
    Upload the full clean/YYYY/MM directory tree to S3.

    Args:
        dir_path: Root clean folder path.

    Returns:
        List of HTTPS URLs uploaded to S3.
    """
    return upload_directory(dir_path, S3_DATASET_BUCKET, prefix="clean")