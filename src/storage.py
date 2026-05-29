"""
File download helper — supports AWS S3 and Cloudinary.
Toggle via USE_CLOUDINARY=true in .env.
"""
import os
import tempfile
from pathlib import Path

USE_CLOUDINARY = os.getenv("USE_CLOUDINARY", "false").lower() == "true"


def download_to_temp(file_key: str, suffix: str) -> str:
    """
    Download a file from S3 or Cloudinary to a local temp file.
    Returns the temp file path. Caller is responsible for deleting it.
    """
    if USE_CLOUDINARY:
        return _download_cloudinary(file_key, suffix)
    return _download_s3(file_key, suffix)


def _download_s3(s3_key: str, suffix: str) -> str:
    import boto3
    from botocore.config import Config

    region = os.getenv("AWS_REGION", "ap-southeast-2")
    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("S3_BUCKET", "resume-screener-uploads")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    s3.download_file(bucket, s3_key, tmp_path)
    return tmp_path


def _cloudinary_resource_type(public_id: str) -> str:
    ext = Path(public_id).suffix.lower()
    if ext in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        return "video"
    return "raw"  # PDFs, DOCXs, and all other non-image files


def _download_cloudinary(public_id: str, suffix: str) -> str:
    import urllib.request

    import cloudinary
    import cloudinary.api

    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )
    resource_type = _cloudinary_resource_type(public_id)
    # Fetch the versioned secure_url from the Cloudinary API — cloudinary_url()
    # omits the version number which causes 401 on raw file downloads.
    resource = cloudinary.api.resource(public_id, resource_type=resource_type)
    secure_url = resource["secure_url"]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    urllib.request.urlretrieve(secure_url, tmp_path)
    return tmp_path
