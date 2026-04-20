import numpy as np

# Weights when a video resume is present (must sum to 1.0)
WEIGHTS_WITH_VIDEO: dict[str, float] = {
    "semantic": 0.40,
    "skill_overlap": 0.20,
    "experience": 0.12,
    "education": 0.08,
    "video": 0.20,
}

# Weights when no video resume — redistribute video weight proportionally (must sum to 1.0)
WEIGHTS_TEXT_ONLY: dict[str, float] = {
    "semantic": 0.50,
    "skill_overlap": 0.25,
    "experience": 0.15,
    "education": 0.10,
}

# Legacy alias used by tests and explainer that expect a single WEIGHTS dict
WEIGHTS = WEIGHTS_WITH_VIDEO

_EDUCATION_RANK: dict[str | None, int] = {
    None: 0, "diploma": 1, "associate": 2,
    "bachelors": 3, "masters": 4, "phd": 5,
}


def get_active_weights(has_video: bool) -> dict[str, float]:
    return WEIGHTS_WITH_VIDEO if has_video else WEIGHTS_TEXT_ONLY


def compute_composite_score(
    semantic: float,
    skill_overlap: float,
    experience: float,
    education: float,
    video: float | None = None,
) -> float:
    """Compute weighted composite score. All inputs should be in [0, 1].
    Pass video=None for text-only candidates to trigger weight redistribution.
    """
    if video is None:
        w = WEIGHTS_TEXT_ONLY
        return (
            w["semantic"] * semantic
            + w["skill_overlap"] * skill_overlap
            + w["experience"] * experience
            + w["education"] * education
        )
    w = WEIGHTS_WITH_VIDEO
    return (
        w["semantic"] * semantic
        + w["skill_overlap"] * skill_overlap
        + w["experience"] * experience
        + w["education"] * education
        + w["video"] * video
    )


def compute_skill_overlap(candidate_skills: list[str], required_skills: list[str]) -> float:
    """Jaccard-style overlap between candidate and required skills."""
    if not required_skills:
        return 0.0
    candidate_set = {s.lower() for s in candidate_skills}
    required_set = {s.lower() for s in required_skills}
    matched = candidate_set & required_set
    return len(matched) / len(required_set)


def compute_experience_score(candidate_years: float, required_years: int) -> float:
    """Normalised experience score; 1.0 at requirement, capped there."""
    if required_years == 0:
        return min(1.0, candidate_years / 5.0)
    return min(1.0, candidate_years / required_years)


def compute_education_score(candidate_level: str | None, required_level: str | None) -> float:
    """Score 1.0 if meets or exceeds requirement, else scaled fraction."""
    c_rank = _EDUCATION_RANK.get(candidate_level, 0)
    r_rank = _EDUCATION_RANK.get(required_level, 0)
    if r_rank == 0:
        return 1.0
    return min(1.0, c_rank / r_rank)
