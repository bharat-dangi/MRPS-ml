"""
24-hour TTL cleanup for raw video files.
Marks video_resumes rows as deleted and removes the S3 objects.
Run via Celery beat every hour, or invoke run_ttl_cleanup() directly.
"""
import logging
import os

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "resume-screener-uploads")


def _delete_s3_object(s3_key: str) -> None:
    import boto3
    from botocore.config import Config
    
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4")
    )
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info("Deleted S3 object: %s", s3_key)
    except Exception:
        logger.warning("Failed to delete S3 object %s", s3_key, exc_info=True)


def mark_expired_videos(db: Session) -> int:
    """
    Find video_resumes older than 24h without deleted_at, mark them,
    and delete their S3 objects. Returns count of rows marked.
    """
    rows = db.execute(
        text(
            "SELECT id, s3_key FROM video_resumes "
            "WHERE deleted_at IS NULL "
            "AND uploaded_at < NOW() - INTERVAL '24 hours'"
        )
    ).fetchall()

    if not rows:
        return 0

    ids = [r[0] for r in rows]
    db.execute(
        text("UPDATE video_resumes SET deleted_at = NOW() WHERE id = ANY(:ids)"),
        {"ids": ids},
    )
    db.commit()

    for _, s3_key in rows:
        _delete_s3_object(s3_key)

    logger.info("TTL cleanup: marked %d video records as deleted", len(ids))
    return len(ids)


def run_ttl_cleanup() -> int:
    """Entry point for Celery beat task — creates its own DB session."""
    from src.db import SessionLocal  # local import to allow standalone use
    with SessionLocal() as db:
        return mark_expired_videos(db)
