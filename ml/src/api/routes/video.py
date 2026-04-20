import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import get_db
from src.models import Candidate, Job, VideoResume
from src.storage import download_to_temp
from src.video.preprocessor import extract_audio
from src.video.transcriber import transcribe
from src.video.audio_features import extract_audio_features
from src.video.video_scorer import compute_video_score

router = APIRouter(prefix="/video", tags=["video"])
logger = logging.getLogger(__name__)


class VideoRequest(BaseModel):
    video_resume_id: int  # DB primary key — ML looks up s3_key from this


class VideoResponse(BaseModel):
    status: str
    video_resume_id: int
    video_score: float
    transcript_similarity: float
    comm_clarity: float
    content_coverage: float


@router.post("/process", response_model=VideoResponse)
def process_video(body: VideoRequest, db: Annotated[Session, Depends(get_db)]) -> VideoResponse:
    """
    1. Look up video_resume record → get s3_key + job_id
    2. Download video from S3
    3. Extract audio → transcribe (Whisper) → audio features (librosa)
    4. Compute video score
    5. Write transcript + score back to DB
    """
    vr = db.scalar(select(VideoResume).where(VideoResume.id == body.video_resume_id))
    if not vr:
        raise HTTPException(status_code=404, detail=f"VideoResume {body.video_resume_id} not found")

    # Determine file extension from s3_key
    ext = "." + vr.s3_key.split(".")[-1] if "." in vr.s3_key else ".mp4"
    video_path = download_to_temp(vr.s3_key, suffix=ext)
    audio_path: str | None = None

    try:
        audio_path = extract_audio(video_path)
        transcript_result = transcribe(audio_path)
        audio_features = extract_audio_features(audio_path)

        # Get job description + required skills for semantic comparison
        job = db.scalar(select(Job).where(Job.id == vr.job_id)) if vr.job_id else None
        jd_text = job.description if job else ""
        jd_skills = job.required_skills if job else []

        word_timestamps = transcript_result.get("word_timestamps")
        score_result = compute_video_score(
            transcript=transcript_result["transcript"],
            jd_text=jd_text,
            audio_features=audio_features,
            jd_skills=jd_skills,
            word_timestamps=word_timestamps,
        )

        # Merge all audio+clarity features into the stored JSON blob
        full_features = {**audio_features, **{
            k: score_result[k]
            for k in ("transcript_sim", "comm_clarity", "content_coverage",
                      "pace_score", "fluency_score", "articulation_score", "filler_rate", "wpm")
            if k in score_result
        }}
        if word_timestamps:
            full_features["word_timestamps"] = word_timestamps

        vr.transcript = transcript_result["transcript"]
        vr.transcript_confidence = transcript_result.get("confidence")
        vr.audio_features = full_features
        vr.video_score = score_result["video_score"]
        vr.process_status = "done"

        # Merge video skill_sources into candidate.skill_sources
        video_sources: dict[str, str] = score_result.get("skill_sources", {})
        if video_sources:
            candidate = db.scalar(select(Candidate).where(Candidate.id == vr.candidate_id))
            if candidate:
                merged = dict(candidate.skill_sources or {})
                merged.update(video_sources)
                candidate.skill_sources = merged

        db.commit()

        logger.info("Video %s processed — score=%.3f", body.video_resume_id, score_result["video_score"])

        return VideoResponse(
            status="done",
            video_resume_id=body.video_resume_id,
            video_score=score_result["video_score"],
            transcript_similarity=score_result["transcript_similarity"],
            comm_clarity=score_result["comm_clarity"],
            content_coverage=score_result["content_coverage"],
        )

    except Exception as exc:
        vr.process_status = "failed"
        db.commit()
        logger.exception("Video processing failed for %s: %s", body.video_resume_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        for path in [video_path, audio_path]:
            if path and os.path.exists(path):
                os.remove(path)


@router.post("/cleanup-ttl")
def cleanup_ttl(db: Annotated[Session, Depends(get_db)]) -> dict:
    """Delete video files older than 24 hours from S3 and mark them in DB."""
    from src.video.ttl_cleaner import mark_expired_videos
    count = mark_expired_videos(db)
    return {"deleted": count}
