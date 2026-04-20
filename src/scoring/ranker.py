import logging
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def rank_all_candidates(job_id: int, engine: Engine) -> None:
    """
    Full ranking pipeline for a job:
    1. Fetch job + all resumes for this job
    2. Embed JD with SBERT
    3. Run pgvector cosine similarity in bulk
    4. Compute composite scores + SHAP
    5. Bulk-upsert screening_results + explainability_data
    """
    # Import here so module can be imported without ML deps installed
    from src.matching.embedder import ResumeEmbedder
    from src.jd_analyzer import analyze_jd
    from src.scoring.composite import (
        WEIGHTS,
        compute_composite_score,
        compute_education_score,
        compute_experience_score,
        compute_skill_overlap,
    )
    from src.scoring.explainer import compute_population_baselines, compute_shap_values

    # Import ORM models via SQLAlchemy; must be importable in worker context
    import sys, os
    _backend = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "backend")
    if os.path.isdir(_backend) and _backend not in sys.path:
        sys.path.insert(0, _backend)

    from app.models import Candidate, ExplainabilityData, Job, Resume, ScreeningResult

    with Session(engine) as session:
        job = session.scalar(select(Job).where(Job.id == job_id))
        if not job:
            logger.error("Job %s not found", job_id)
            return

        jd_analysis = analyze_jd(job.description)
        embedder = ResumeEmbedder()
        jd_embedding = embedder.encode(job.description)

        # Fetch all candidates with resumes for this job
        rows = session.execute(
            select(Candidate, Resume)
            .join(Resume, Resume.candidate_id == Candidate.id)
            .where(Resume.job_id == job_id, Candidate.embedding.is_not(None))
        ).all()

        if not rows:
            logger.warning("No embedded candidates found for job %s", job_id)
            return

        candidates = [row[0] for row in rows]

        # Compute per-feature scores
        feature_list: list[dict[str, float]] = []
        for candidate in candidates:
            c_emb = np.array(candidate.embedding, dtype=np.float32)
            semantic = float(np.dot(jd_embedding, c_emb))  # normalised → cosine

            skill_overlap = compute_skill_overlap(
                candidate.skills or [],
                jd_analysis["required_skills"],
            )
            exp_score = compute_experience_score(
                candidate.years_experience or 0.0,
                jd_analysis["min_experience"],
            )
            edu_score = compute_education_score(
                candidate.education_level,
                jd_analysis["education_requirement"],
            )

            # Video score from most recent video resume for this job
            video_score = 0.0
            from app.models import VideoResume
            vr = session.scalar(
                select(VideoResume)
                .where(VideoResume.candidate_id == candidate.id, VideoResume.job_id == job_id)
                .order_by(VideoResume.uploaded_at.desc())
            )
            if vr and vr.video_score is not None:
                video_score = vr.video_score

            feature_list.append({
                "semantic": max(0.0, semantic),
                "skill_overlap": skill_overlap,
                "experience": exp_score,
                "education": edu_score,
                "video": video_score,
            })

        baselines = compute_population_baselines(feature_list)

        # Compute composite scores and rank
        scored = [
            (candidate, features, compute_composite_score(**features))
            for candidate, features in zip(candidates, feature_list)
        ]
        scored.sort(key=lambda x: x[2], reverse=True)

        # Bulk-upsert results
        for rank, (candidate, features, composite_score) in enumerate(scored, start=1):
            shap = compute_shap_values(features, baselines)

            existing = session.scalar(
                select(ScreeningResult).where(
                    ScreeningResult.job_id == job_id,
                    ScreeningResult.candidate_id == candidate.id,
                )
            )
            if existing:
                sr = existing
                sr.composite_score = composite_score
                sr.semantic_score = features["semantic"]
                sr.skill_overlap_score = features["skill_overlap"]
                sr.experience_score = features["experience"]
                sr.education_score = features["education"]
                sr.video_score = features["video"]
                sr.rank = rank
            else:
                sr = ScreeningResult(
                    job_id=job_id,
                    candidate_id=candidate.id,
                    composite_score=composite_score,
                    semantic_score=features["semantic"],
                    skill_overlap_score=features["skill_overlap"],
                    experience_score=features["experience"],
                    education_score=features["education"],
                    video_score=features["video"],
                    rank=rank,
                )
                session.add(sr)
                session.flush()

            # Upsert explainability
            if sr.explainability:
                sr.explainability.shap_values = shap
                sr.explainability.baseline_score = sum(
                    WEIGHTS[k] * baselines[k] for k in WEIGHTS
                )
            else:
                session.add(ExplainabilityData(
                    screening_result_id=sr.id,
                    shap_values=shap,
                    baseline_score=sum(WEIGHTS[k] * baselines[k] for k in WEIGHTS),
                ))

        session.commit()
        logger.info("Ranked %d candidates for job %s", len(scored), job_id)
