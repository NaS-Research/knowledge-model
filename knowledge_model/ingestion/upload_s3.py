"""
upload_s3.py
------------
Provides functions to upload files to AWS S3.
"""

import os
import boto3

S3_PDF_BUCKET = os.getenv("S3_PDF_BUCKET", "nas-knowledge-model-pdfs")
S3_DATASET_BUCKET = os.getenv("S3_DATASET_BUCKET", "nas-knowledge-model-dataset")

s3_client = boto3.client("s3")

def upload_pdf_to_s3(file_path: str) -> str:
    """
    Upload a PDF file to the S3 bucket for PDFs.
    
    :param file_path: Local file path to the PDF.
    :return: The S3 URL of the uploaded file.
    """
    filename = os.path.basename(file_path)
    s3_key = filename 
    s3_client.upload_file(file_path, S3_PDF_BUCKET, s3_key)
    return f"https://{S3_PDF_BUCKET}.s3.amazonaws.com/{s3_key}"

def upload_dataset_to_s3(file_path: str) -> str:
    """
    Upload a dataset file (e.g. JSONL) to the S3 bucket for processed data.
    
    :param file_path: Local file path to the dataset.
    :return: The S3 URL of the uploaded file.
    """
    filename = os.path.basename(file_path)
    s3_key = filename 
    s3_client.upload_file(file_path, S3_DATASET_BUCKET, s3_key)
    return f"https://{S3_DATASET_BUCKET}.s3.amazonaws.com/{s3_key}"