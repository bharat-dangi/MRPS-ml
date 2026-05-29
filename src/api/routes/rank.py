import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import get_db
from src.jd_analyzer import analyze_jd
from src.matching.bm25_scorer import BM25Scorer
from src.matching.embedder import ResumeEmbedder
from src.models import Candidate, ExplainabilityData, Job, Resume, ScreeningResult, VideoResume
from src.scoring.composite import (
    compute_composite_score,
    compute_education_score,
    compute_experience_score,
    compute_skill_overlap,
    get_active_weights,
)
from src.scoring.explainer import compute_population_baselines, compute_shap_values

router = APIRouter(prefix="/rank", tags=["rank"])
logger = logging.getLogger(__name__)


class RankRequest(BaseModel):
    job_id: int


class RankResponse(BaseModel):
    status: str
    job_id: int
    ranked_count: int


def _score_candidate(
    candidate: Candidate,
    cosine_norm: float,
    bm25_norm: float,
    jd_analysis: dict,
    job_id: int,
    db: Session,
) -> tuple[dict, bool]:
    """Compute all features for one candidate; returns (features, has_video).

    cosine_norm and bm25_norm are both pre-normalised to [0,1] relative to the
    batch maximum, so they are on the same scale before blending.
    """
    hybrid_semantic = 0.70 * cosine_norm + 0.30 * bm25_norm

    skill_overlap = compute_skill_overlap(candidate.skills or [], jd_analysis["required_skills"])
    exp_score = compute_experience_score(candidate.years_experience or 0.0, jd_analysis["min_experience"])
    edu_score = compute_education_score(candidate.education_level, jd_analysis["education_requirement"])

    vr = db.scalar(
        select(VideoResume)
        .where(
            VideoResume.candidate_id == candidate.id,
            VideoResume.job_id == job_id,
            VideoResume.deleted_at.is_(None),
        )
        .order_by(VideoResume.id.desc())
    )
    has_video = vr is not None and vr.video_score is not None
    # Use None (not 0.0) so compute_composite_score picks WEIGHTS_TEXT_ONLY
    video_score = float(vr.video_score) if has_video else None

    return {
        "semantic": hybrid_semantic,
        "skill_overlap": skill_overlap,
        "experience": exp_score,
        "education": edu_score,
        "video": video_score,
    }, has_video


def _upsert_screening_result(
    job_id: int, candidate: Candidate, composite: float, features: dict,
    rank: int, db: Session,
) -> ScreeningResult:
    existing = db.scalar(
        select(ScreeningResult).where(
            ScreeningResult.job_id == job_id,
            ScreeningResult.candidate_id == candidate.id,
        )
    )
    if existing:
        existing.composite_score = composite
        existing.semantic_score = features["semantic"]
        existing.skill_overlap_score = features["skill_overlap"]
        existing.experience_score = features["experience"]
        existing.education_score = features["education"]
        existing.video_score = features["video"] or 0.0
        existing.rank = rank
        return existing

    sr = ScreeningResult(
        job_id=job_id,
        candidate_id=candidate.id,
        composite_score=composite,
        semantic_score=features["semantic"],
        skill_overlap_score=features["skill_overlap"],
        experience_score=features["experience"],
        education_score=features["education"],
        video_score=features["video"] or 0.0,
        rank=rank,
    )
    db.add(sr)
    db.flush()
    return sr


def _upsert_explainability(sr: ScreeningResult, shap: dict, baseline_score: float, db: Session) -> None:
    existing_xai = db.scalar(
        select(ExplainabilityData).where(ExplainabilityData.screening_result_id == sr.id)
    )
    if existing_xai:
        existing_xai.shap_values = shap
        existing_xai.baseline_score = baseline_score
    else:
        db.add(ExplainabilityData(
            screening_result_id=sr.id,
            shap_values=shap,
            baseline_score=baseline_score,
        ))


@router.post("/{job_id}", responses={404: {"description": "Job not found"}})
def rank_job(job_id: int, db: Annotated[Session, Depends(get_db)]) -> RankResponse:
    """
    Full ranking pipeline for one job:
    1. Read job + all submitted candidates (with embeddings) from DB
    2. Embed JD with SBERT + build BM25 index for hybrid semantic
    3. Score each candidate (hybrid_semantic + skill_overlap + experience + education + video)
    4. Compute SHAP values
    5. Write screening_results + explainability_data back to DB
    """
    job = db.scalar(select(Job).where(Job.id == job_id))
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Candidates linked to this job via either modality — Resume OR VideoResume.
    resume_cids = select(Resume.candidate_id).where(Resume.job_id == job_id)
    video_cids = select(VideoResume.candidate_id).where(
        VideoResume.job_id == job_id,
        VideoResume.deleted_at.is_(None),
    )
    rows = db.execute(
        select(Candidate)
        .where(
            Candidate.id.in_(resume_cids) | Candidate.id.in_(video_cids),
            Candidate.embedding.is_not(None),
        )
    ).all()

    if not rows:
        logger.warning("No embedded candidates for job %s", job_id)
        return RankResponse(status="no_candidates", job_id=job_id, ranked_count=0)

    jd_analysis = analyze_jd(job.description)
    embedder = ResumeEmbedder()
    jd_emb = embedder.encode(job.description)

    candidates = [r[0] for r in rows]

    # Batch-encode all candidates for efficiency
    candidate_vectors = embedder.encode_batch([c.raw_text or "" for c in candidates])

    # Cosine similarity: embeddings are L2-normalised → dot product == cosine
    cosine_sims = candidate_vectors.dot(jd_emb).clip(min=0.0)
    cosine_max = float(cosine_sims.max()) if cosine_sims.size > 0 else 1.0

    bm25 = BM25Scorer()
    bm25.fit([c.raw_text or "" for c in candidates])
    bm25_scores = bm25.score(job.description)
    bm25_max = float(bm25_scores.max()) if bm25_scores.size > 0 else 1.0

    # Merge user-specified required_skills with JD-parsed ones so both count
    merged_required = list(dict.fromkeys(
        jd_analysis["required_skills"] + [s.lower() for s in (job.required_skills or [])]
    ))
    jd_analysis = {**jd_analysis, "required_skills": merged_required}

    feature_list: list[dict] = []
    for idx, candidate in enumerate(candidates):
        cosine_norm = cosine_sims[idx] / cosine_max if cosine_max > 0 else 0.0
        bm25_norm = float(bm25_scores[idx]) / bm25_max if bm25_max > 0 else 0.0
        features, _ = _score_candidate(
            candidate,
            float(cosine_norm),
            bm25_norm,
            jd_analysis,
            job_id,
            db,
        )
        feature_list.append(features)

    baselines = compute_population_baselines(feature_list)
    scored = sorted(
        zip(candidates, feature_list, strict=False),
        key=lambda x: compute_composite_score(**x[1]),
        reverse=True,
    )

    for rank, (candidate, features) in enumerate(scored, start=1):
        composite = compute_composite_score(**features)
        shap = compute_shap_values(features, baselines)
        sr = _upsert_screening_result(job_id, candidate, composite, features, rank, db)
        weights = get_active_weights(has_video=features["video"] is not None)
        baseline_score = sum(weights.get(k, 0) * baselines.get(k, 0) for k in weights)
        _upsert_explainability(sr, shap, baseline_score, db)

    db.commit()
    logger.info("Ranked %d candidates for job %s", len(scored), job_id)
    return RankResponse(status="done", job_id=job_id, ranked_count=len(scored))
