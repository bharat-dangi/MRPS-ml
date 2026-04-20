"""Unit tests for composite scoring and SHAP — no DB or ML models needed."""
from src.scoring.composite import (
    WEIGHTS,
    compute_composite_score,
    compute_education_score,
    compute_experience_score,
    compute_skill_overlap,
)
from src.scoring.explainer import compute_population_baselines, compute_shap_values


def test_weights_sum_to_one() -> None:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_perfect_score() -> None:
    score = compute_composite_score(
        semantic=1.0, skill_overlap=1.0, experience=1.0, education=1.0, video=1.0
    )
    assert abs(score - 1.0) < 1e-9


def test_no_video_score() -> None:
    score = compute_composite_score(
        semantic=1.0, skill_overlap=1.0, experience=1.0, education=1.0, video=0.0
    )
    assert abs(score - 0.80) < 1e-9


def test_skill_overlap_exact_match() -> None:
    assert compute_skill_overlap(["python", "docker"], ["python", "docker"]) == 1.0


def test_skill_overlap_partial() -> None:
    overlap = compute_skill_overlap(["python"], ["python", "docker", "fastapi"])
    assert abs(overlap - 1 / 3) < 1e-9


def test_skill_overlap_no_required() -> None:
    assert compute_skill_overlap(["python"], []) == 0.0


def test_experience_score_meets_requirement() -> None:
    assert compute_experience_score(5.0, 3) == 1.0


def test_experience_score_below_requirement() -> None:
    score = compute_experience_score(1.0, 4)
    assert 0.0 < score < 1.0


def test_education_score_exceeds_requirement() -> None:
    assert compute_education_score("phd", "masters") == 1.0


def test_education_score_below_requirement() -> None:
    score = compute_education_score("bachelors", "phd")
    assert score < 1.0


def test_shap_values_sum_to_score_minus_baseline() -> None:
    features = {"semantic": 0.8, "skill_overlap": 0.6, "experience": 0.7, "education": 0.9, "video": 0.5}
    baselines = {"semantic": 0.5, "skill_overlap": 0.5, "experience": 0.5, "education": 0.5, "video": 0.5}
    shap = compute_shap_values(features, baselines)
    shap_sum = sum(shap.values())
    score = compute_composite_score(**features)
    baseline_score = compute_composite_score(**baselines)
    assert abs(shap_sum - (score - baseline_score)) < 1e-9


def test_demographic_parity_within_threshold() -> None:
    """Both groups drawn from same distribution — gap must be < 5%."""
    import random
    rng = random.Random(42)

    def group_scores(seed: int) -> list[float]:
        r = random.Random(seed)
        return [
            compute_composite_score(
                semantic=max(0.0, min(1.0, r.gauss(0.6, 0.15))),
                skill_overlap=max(0.0, min(1.0, r.gauss(0.5, 0.2))),
                experience=max(0.0, min(1.0, r.gauss(0.55, 0.18))),
                education=max(0.0, min(1.0, r.gauss(0.65, 0.12))),
                video=max(0.0, min(1.0, r.gauss(0.55, 0.15))),
            )
            for _ in range(50)
        ]

    a, b = group_scores(42), group_scores(99)
    gap = abs(sum(a) / len(a) - sum(b) / len(b))
    assert gap < 0.05, f"Demographic parity gap {gap:.4f} exceeds threshold 0.05"
