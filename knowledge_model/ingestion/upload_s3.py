"""
upload_s3.py
------------
Reusable helpers for uploading artifacts to AWS S3.

• PDFs             → nas-knowledge-model-pdfs/<year>/<month>/<filename>
• train.jsonl      → nas-knowledge-model-dataset/<prefix>/YYYY/MM/train.jsonl
• clean chunks     → nas-knowledge-model-dataset/clean/YYYY/MM/<filename>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, UTC as _UTC

import boto3
import threading
try:
    from tqdm.auto import tqdm  # progress bar
except ImportError:  # fallback if tqdm isn't installed
    tqdm = None

class _TqdmCallback:
    """
    Streaming callback for boto3 that renders a progress bar with tqdm.
 
    It is created once per file upload and fed the byte‐count chunks that
    boto3 transfers.
    """
    def __init__(self, filename: Path):
        self._filename = filename
        self._size = filename.stat().st_size
        self._seen = 0
        self._lock = threading.Lock()
        self._bar = (
            tqdm(total=self._size, unit="B", unit_scale=True,
                 unit_divisor=1024, desc=filename.name, leave=False)
            if tqdm is not None else None
        )
 
    def __call__(self, bytes_amount: int) -> None:
        with self._lock:
            self._seen += bytes_amount
            if self._bar:
                self._bar.update(bytes_amount)
            # fall‑back: basic percentage if tqdm is missing
            elif self._size:
                pct = (self._seen / self._size) * 100
                print(f"\rUploading {self._filename.name}: {pct:5.1f}% ", end="", flush=True)
 
    def close(self) -> None:
        if self._bar:
            self._bar.close()

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
    cb = _TqdmCallback(path)
    s3_client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs=extra_args or {},
        Callback=cb,
    )
    cb.close()
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


def upload_pdf_to_s3(
    file_path: str,
    *,
    year: str | None = None,
    month: str | None = None,
) -> str:
    """
    Upload a single PDF to nas-knowledge-model-pdfs/year/month/filename.

    If year/month are omitted, current UTC YYYY/MM are used.

    Returns the HTTPS URL of the uploaded PDF.
    """
    if year is None or month is None:
        now = datetime.now(_UTC)
        year = year or f"{now.year:04d}"
        month = month or f"{now.month:02d}"

    filename = os.path.basename(file_path)
    key = f"{year}/{month}/{filename}"
    return _upload_file(file_path, S3_PDF_BUCKET, key)


def upload_dataset_to_s3(
    file_path: str,
    *,
    year: str | None = None,
    month: str | None = None,
    prefix: str = "",
) -> str:
    """
    Upload a dataset file into nas-knowledge-model-dataset/{prefix}/YYYY/MM/filename.

    If year/month omitted, uses current UTC date. `prefix` can be "" or
    something like "raw" or "combined".
    """
    if year is None or month is None:
        now = datetime.now(_UTC)
        year = year or f"{now.year:04d}"
        month = month or f"{now.month:02d}"

    filename = os.path.basename(file_path)
    key_prefix = f"{prefix}/" if prefix else ""
    key = f"{key_prefix}{year}/{month}/{filename}"
    return _upload_file(file_path, S3_DATASET_BUCKET, key)


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