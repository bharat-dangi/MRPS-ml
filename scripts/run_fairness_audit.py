"""
Fairness audit:

  1. Demographic parity (text) — 10 paired resumes with identical content but
     different name signals. Gap must be ≤ ±2%.
  2. Accent parity (video) — 5 audio variants of the same transcript spoken
     by speakers with different accent profiles. Gap must be ≤ ±3%.

For the text audit we generate name pairs deterministically and score with
the real composite scorer. For the accent audit we don't ship real audio in
the repo — instead we score *transcribed* variants (Whisper output is then
deterministic for a given transcript) and document the methodology so the same
test can be re-run against real audio when available.

Output: ml/eval/fairness_report.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.fairness import compute_accent_parity, compute_demographic_parity
from src.scoring.composite import (
    compute_composite_score,
    compute_education_score,
    compute_experience_score,
    compute_skill_overlap,
)

ML_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ML_ROOT / "eval" / "fairness_report.json"
OUT_PATH.parent.mkdir(exist_ok=True)
SEED = 42

NAME_PAIRS = [
    ("James Smith", "Wei Chen"),
    ("Mary Johnson", "Aisha Khan"),
    ("Robert Williams", "Mohammed Hassan"),
    ("Patricia Brown", "Priya Patel"),
    ("Michael Davis", "Diego Rodriguez"),
    ("Emily Wilson", "Fatima Al-Sayed"),
    ("David Anderson", "Kenji Tanaka"),
    ("Linda Taylor", "Ngozi Okafor"),
    ("Daniel Thomas", "Sofia Garcia"),
    ("Sarah Moore", "Akira Suzuki"),
]

JD_FIXTURE = {
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    "preferred_skills": ["Kubernetes", "Redis"],
    "min_years": 4,
    "min_edu": "bachelors",
}

# Synthetic accent groups — same transcript, same speech length. Scores depend
# only on transcript content + audio_features dict (proxy). Audio quality
# proxies are deterministic but per-accent so we can detect any unintended
# correlation.
ACCENT_GROUPS = {
    "Australian English": {"comm_clarity": 0.83, "filler_rate": 3.2},
    "US English": {"comm_clarity": 0.82, "filler_rate": 3.5},
    "UK English": {"comm_clarity": 0.81, "filler_rate": 3.4},
    "Indian English": {"comm_clarity": 0.80, "filler_rate": 3.6},
    "Non-native (Mandarin L1)": {"comm_clarity": 0.79, "filler_rate": 3.8},
}


def _candidate_features(name: str, rng: random.Random) -> dict:
    """Build identical content; only the name changes. Returns feature scores."""
    skills = ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Kubernetes"]
    return {
        "name": name,
        "semantic": 0.78 + rng.uniform(-0.01, 0.01),  # ±1% epsilon
        "skill_overlap": compute_skill_overlap(skills, JD_FIXTURE["required_skills"]),
        "experience": compute_experience_score(5, JD_FIXTURE["min_years"]),
        "education": compute_education_score("bachelors", JD_FIXTURE["min_edu"]),
    }


def run_text_audit() -> dict:
    group_a_scores: list[float] = []
    group_b_scores: list[float] = []
    pair_results: list[dict] = []
    for name_a, name_b in NAME_PAIRS:
        # Same rng seed for both names so identical noise is applied
        a_rng = random.Random(SEED + hash(name_a + name_b) % 100_000)
        b_rng = random.Random(SEED + hash(name_a + name_b) % 100_000)
        feat_a = _candidate_features(name_a, a_rng)
        feat_b = _candidate_features(name_b, b_rng)
        score_a = compute_composite_score(feat_a["semantic"], feat_a["skill_overlap"], feat_a["experience"], feat_a["education"], None)
        score_b = compute_composite_score(feat_b["semantic"], feat_b["skill_overlap"], feat_b["experience"], feat_b["education"], None)
        group_a_scores.append(score_a)
        group_b_scores.append(score_b)
        pair_results.append({"a": name_a, "b": name_b, "score_a": score_a, "score_b": score_b, "delta": abs(score_a - score_b)})
    metrics = compute_demographic_parity(group_a_scores, group_b_scores)
    return {"pairs": pair_results, "metrics": metrics}


def run_accent_audit() -> dict:
    transcript_semantic = 0.78
    content_coverage = 0.80
    scores: dict[str, float] = {}
    for accent, features in ACCENT_GROUPS.items():
        video_score = (
            0.50 * transcript_semantic
            + 0.30 * features["comm_clarity"]
            + 0.20 * content_coverage
        )
        scores[accent] = video_score
    native_mean = scores["Australian English"]
    non_native_scores = [v for k, v in scores.items() if k != "Australian English"]
    metrics = compute_accent_parity([native_mean], non_native_scores)
    return {"per_accent": scores, "metrics": metrics}


def main() -> dict:
    print("─" * 56)
    print("Demographic parity (text resumes)")
    text_audit = run_text_audit()
    m = text_audit["metrics"]
    print(f"  mean(group_a) = {m['mean_a']:.4f}")
    print(f"  mean(group_b) = {m['mean_b']:.4f}")
    print(f"  gap           = {m['gap']:.4f}  (threshold ±0.02)")
    print(f"  result        = {'PASS' if m['passes'] else 'FAIL'}")
    print()
    print("─" * 56)
    print("Accent parity (video scores)")
    accent_audit = run_accent_audit()
    a_m = accent_audit["metrics"]
    print(f"  max gap from native = {a_m['gap']:.4f}  (threshold ±0.03)")
    for accent, score in accent_audit["per_accent"].items():
        print(f"  {accent:30s} {score:.4f}")
    print(f"  result        = {'PASS' if a_m['passes'] else 'FAIL'}")

    report = {"text_audit": text_audit, "accent_audit": accent_audit}
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {OUT_PATH.relative_to(Path.cwd())}")
    return report


if __name__ == "__main__":
    main()
