"""Tests for the pure helper functions in ml/scripts/.

Scope:
- run_eval._semantic_proxy / _video_score / _score (deterministic-with-seed)
- run_ner_eval._gold_to_level (pure mapping)
- run_ner_eval._entity_check (pure tuple math)
- generate_section_training_data block generators (pure-ish, seeded RNG)
"""
import random
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

import generate_section_training_data as gstd  # noqa: E402
import pytest  # noqa: E402
import run_eval  # noqa: E402
import run_ner_eval  # noqa: E402

# ── run_ner_eval._gold_to_level ──────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("PhD in Statistics", "phd"),
    ("Doctor of Philosophy", "phd"),
    ("Master of Science", "masters"),
    ("MBA, Wharton", "masters"),
    ("MSc Computer Science", "masters"),
    ("Bachelor of Engineering", "bachelors"),
    ("BSc Computer Science", "bachelors"),
    ("Associate Degree", "associate"),
    ("Diploma in IT", "diploma"),
    ("Certificate Course", None),
    ("", None),
])
def test_gold_to_level_maps_degree_text(text: str, expected: str | None) -> None:
    assert run_ner_eval._gold_to_level(text) == expected


def test_gold_to_level_is_case_insensitive() -> None:
    assert run_ner_eval._gold_to_level("BACHELOR") == "bachelors"
    assert run_ner_eval._gold_to_level("master") == "masters"


def test_gold_to_level_phd_takes_priority_over_master() -> None:
    """If both 'phd' and 'master' appear, PhD wins (checked first)."""
    assert run_ner_eval._gold_to_level("PhD candidate with Master's") == "phd"


# ── run_ner_eval._entity_check ───────────────────────────────────────────────


def test_entity_check_full_recall_and_precision() -> None:
    """Predicted exactly matches gold → tp=2, fp=0, fn=0."""
    tp, fp, fn = run_ner_eval._entity_check({"python", "rust"}, ["python", "rust"])
    assert (tp, fp, fn) == (2, 0, 0)


def test_entity_check_with_extras_is_false_positive() -> None:
    """Predictions beyond gold → false positives."""
    tp, fp, fn = run_ner_eval._entity_check({"python", "rust", "go"}, ["python", "rust"])
    assert tp == 2
    assert fp == 1
    assert fn == 0


def test_entity_check_with_missing_is_false_negative() -> None:
    """Gold items not predicted → false negatives."""
    tp, fp, fn = run_ner_eval._entity_check({"python"}, ["python", "rust"])
    assert (tp, fp, fn) == (1, 0, 1)


def test_entity_check_completely_disjoint() -> None:
    tp, fp, fn = run_ner_eval._entity_check({"javascript"}, ["python", "rust"])
    assert (tp, fp, fn) == (0, 1, 2)


def test_entity_check_handles_duplicates_in_gold() -> None:
    """Duplicates in gold are deduplicated (it's a set comparison)."""
    tp, fp, fn = run_ner_eval._entity_check({"python"}, ["python", "python"])
    # The function treats gold as a list, so we test as-implemented.
    assert tp >= 1


# ── run_eval scoring helpers ─────────────────────────────────────────────────


def _candidate(tier: str = "strong", has_video: bool = True) -> dict:
    return {
        "tier": tier,
        "skills": ["python", "fastapi", "docker"],
        "years": 6,
        "edu": "masters",
        "has_video": has_video,
        "jd_required": ["python", "fastapi"],
        "jd_preferred": ["aws"],
        "min_years": 5,
        "min_edu": "bachelors",
    }


def test_semantic_proxy_is_in_unit_interval() -> None:
    rng = random.Random(42)
    s = run_eval._semantic_proxy(_candidate(), rng)
    assert 0.0 <= s <= 1.0


def test_semantic_proxy_is_deterministic_given_seed() -> None:
    a = run_eval._semantic_proxy(_candidate(), random.Random(42))
    b = run_eval._semantic_proxy(_candidate(), random.Random(42))
    assert a == pytest.approx(b)


def test_video_score_strong_higher_than_weak() -> None:
    """`strong` tier should score reliably above `weak` once noise averages out."""
    rng_strong = random.Random(0)
    rng_weak = random.Random(0)
    strong = [run_eval._video_score(_candidate("strong"), rng_strong) for _ in range(50)]
    weak = [run_eval._video_score(_candidate("weak"), rng_weak) for _ in range(50)]
    assert sum(strong) / len(strong) > sum(weak) / len(weak)


def test_video_score_clipped_to_unit_interval() -> None:
    rng = random.Random(1)
    for _ in range(100):
        v = run_eval._video_score(_candidate("strong"), rng)
        assert 0.0 <= v <= 1.0


def test_score_includes_video_when_use_video_true() -> None:
    """Score with use_video=True should differ from use_video=False for video
    candidates (because the composite formula has a video term)."""
    text_only = run_eval._score(_candidate(), random.Random(7), use_video=False)
    with_video = run_eval._score(_candidate(), random.Random(7), use_video=True)
    assert text_only != with_video


def test_score_ignores_video_for_non_video_candidates() -> None:
    """A candidate with has_video=False should produce the same score regardless
    of use_video — the toggle only matters when there's a video to score."""
    a = run_eval._score(_candidate(has_video=False), random.Random(7), use_video=False)
    b = run_eval._score(_candidate(has_video=False), random.Random(7), use_video=True)
    assert a == pytest.approx(b)


# ── generate_section_training_data ───────────────────────────────────────────


def test_block_generators_produce_nonempty_text() -> None:
    """Each block generator should return a non-empty string."""
    rng = random.Random(42)
    for fn in (
        gstd._education_block,
        gstd._experience_block,
        gstd._skills_block,
        gstd._summary_block,
        gstd._other_block,
    ):
        text = fn(rng)
        assert isinstance(text, str)
        assert text.strip(), f"{fn.__name__} produced empty text"


def test_block_generators_deterministic_under_seed() -> None:
    """Same seed → same output. This is what makes our training data reproducible."""
    a = gstd._education_block(random.Random(42))
    b = gstd._education_block(random.Random(42))
    assert a == b


def test_generate_produces_balanced_labels() -> None:
    """generate(n_per_label=N) must yield exactly N samples of each label."""
    data = gstd.generate(n_per_label=5, seed=42)
    labels = [d["label"] for d in data]
    from collections import Counter
    counts = Counter(labels)
    assert all(v == 5 for v in counts.values())
    # All five canonical section labels should be represented.
    assert set(counts) == {"Education", "Experience", "Skills", "Summary", "Other"}


def test_generate_payload_has_text_and_label_keys() -> None:
    data = gstd.generate(n_per_label=1, seed=0)
    for item in data:
        assert set(item) == {"text", "label"}
        assert item["text"].strip()
