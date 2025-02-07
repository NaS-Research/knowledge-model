"""
upload_s3.py
Handles uploading PDFs to an S3 bucket using credentials from environment variables.
"""

import os
import boto3
import logging

logger = logging.getLogger(__name__)

def upload_pdf_to_s3(local_pdf_path: str) -> str:
    """
    Uploads a local PDF to S3 and returns the s3:// URL (or https:// style if you prefer).
    Requires environment variables:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_DEFAULT_REGION
      - S3_BUCKET_NAME
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    bucket = os.getenv("S3_BUCKET_NAME", "default-bucket")
    key = os.path.basename(local_pdf_path)  
    logger.info("Uploading %s to s3://%s/%s", local_pdf_path, bucket, key)

    s3.upload_file(local_pdf_path, bucket, key)

    s3_url = f"s3://{bucket}/{key}"
    return s3_url
