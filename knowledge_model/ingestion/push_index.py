"""
push_index.py
-------------
Build (optional) and upload the FAISS index to S3.

Usage:
  python -m knowledge_model.ingestion.push_index \
      --src data/combined/combined_v2.jsonl \
      --outdir data/faiss \
      --s3  s3://nas-assets/faiss
"""

from __future__ import annotations
import argparse
from pathlib import Path
from knowledge_model.ingestion.build_faiss import build_faiss_index
from knowledge_model.ingestion.upload_s3 import upload_directory

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src",    required=True, type=Path, help="JSONL or dir")
    p.add_argument("--outdir", required=True, type=Path, help="Local output dir")
    p.add_argument("--s3",     required=True, type=str,  help="s3://bucket/prefix")
    p.add_argument("--model",  default="all-MiniLM-L6-v2")
    p.add_argument("--field",  default="text")
    args = p.parse_args()

    # ① Build / refresh local index
    build_faiss_index(args.src, args.outdir, args.model, field=args.field)

    # ② Upload two files in that directory
    bucket, *prefix_parts = args.s3.replace("s3://", "").split("/", 1)
    prefix = prefix_parts[0] if prefix_parts else ""
    urls = upload_directory(args.outdir, bucket=bucket, prefix=prefix, recurse=False)

    print("Uploaded:\n" + "\n".join(urls))

if __name__ == "__main__":
    main()