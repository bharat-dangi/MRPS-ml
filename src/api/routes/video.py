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
from src.video.audio_features import extract_audio_features
from src.video.preprocessor import extract_audio
from src.video.transcriber import transcribe
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


# Treated as "unknown" so the transcript NER can overwrite the candidate's name.
_PLACEHOLDER_NAMES = {"test", "unknown", "demo", "candidate", "n/a", "na", "-"}


def _is_placeholder_name(name: str | None) -> bool:
    if not name:
        return True
    stripped = name.strip()
    if not stripped:
        return True
    return stripped.lower() in _PLACEHOLDER_NAMES


def _apply_text_and_embedding(candidate: Candidate, transcript_text: str) -> None:
    if not (candidate.raw_text or "").strip():
        candidate.raw_text = transcript_text
    if candidate.embedding is None:
        from src.matching.embedder import ResumeEmbedder
        candidate.embedding = ResumeEmbedder().encode(transcript_text)


def _apply_profile_scalars(candidate: Candidate, profile) -> None:
    if _is_placeholder_name(candidate.full_name) and profile.full_name != "Unknown":
        candidate.full_name = profile.full_name
    if not (candidate.email or "").strip() and profile.email:
        candidate.email = profile.email
    if not (candidate.phone or "").strip() and profile.phone:
        candidate.phone = profile.phone
    if (candidate.years_experience or 0) == 0 and profile.years_experience > 0:
        candidate.years_experience = profile.years_experience
    if not candidate.education_level and profile.education_level:
        candidate.education_level = profile.education_level


def _merge_skills(
    candidate: Candidate,
    ner_skills: list[str],
    jd_skill_sources: dict[str, str],
) -> None:
    merged_video_sources = {**dict.fromkeys(ner_skills, "video"), **jd_skill_sources}
    if not merged_video_sources:
        return
    merged_sources = dict(candidate.skill_sources or {})
    merged_sources.update(merged_video_sources)
    candidate.skill_sources = merged_sources
    existing_skills = {s.lower() for s in (candidate.skills or [])}
    new_skills = [s for s in merged_video_sources if s.lower() not in existing_skills]
    if new_skills:
        candidate.skills = list(candidate.skills or []) + new_skills


def _backfill_candidate_from_video(
    candidate: Candidate,
    transcript_text: str,
    jd_skill_sources: dict[str, str],
) -> None:
    """Hydrate empty candidate fields from the transcript so video-only uploads can be ranked."""
    if not transcript_text:
        return
    # Deferred — pulls spaCy + 5k-row taxonomy.
    from src.ner.extractor import extract_candidate_profile
    profile = extract_candidate_profile(transcript_text)
    _apply_text_and_embedding(candidate, transcript_text)
    _apply_profile_scalars(candidate, profile)
    _merge_skills(candidate, profile.skills, jd_skill_sources)


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

        candidate = db.scalar(select(Candidate).where(Candidate.id == vr.candidate_id))
        if candidate:
            _backfill_candidate_from_video(
                candidate,
                (transcript_result.get("transcript") or "").strip(),
                score_result.get("skill_sources", {}),
            )

        db.commit()

        logger.info("Video %s processed — score=%.3f", body.video_resume_id, score_result["video_score"])

        return VideoResponse(
            status="done",
            video_resume_id=body.video_resume_id,
            video_score=score_result["video_score"],
            transcript_similarity=score_result["transcript_sim"],
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
