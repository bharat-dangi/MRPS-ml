"""Tests for jd_analyzer.analyze_jd — parses JD text into structured fields.

Requires spaCy + en_core_web_sm. Each test feeds a tiny JD and asserts on the
returned shape and key extractions.
"""

from src.jd_analyzer import analyze_jd

# ── shape ────────────────────────────────────────────────────────────────────


def test_analyze_jd_returns_all_expected_keys() -> None:
    out = analyze_jd("We need a Python engineer with 3 years of experience.")
    assert set(out) == {
        "required_skills", "preferred_skills", "min_experience", "education_requirement",
    }


# ── min_experience extraction ────────────────────────────────────────────────


def test_min_experience_extracts_smallest_year_value() -> None:
    """Picks the *minimum* year requirement — JDs often state ranges."""
    out = analyze_jd("3+ years required, 5 years preferred.")
    assert out["min_experience"] == 3


def test_min_experience_is_zero_when_no_years_mentioned() -> None:
    out = analyze_jd("We want a great engineer.")
    assert out["min_experience"] == 0


def test_min_experience_handles_word_variations() -> None:
    """`year` and `years` both match the regex."""
    out = analyze_jd("Minimum 7 year of experience required.")
    assert out["min_experience"] == 7


# ── education_requirement extraction ─────────────────────────────────────────


def test_education_bachelor_keyword() -> None:
    out = analyze_jd("Bachelor's degree in Computer Science required.")
    assert out["education_requirement"] == "bachelors"


def test_education_master_keyword() -> None:
    out = analyze_jd("Master's or MBA preferred.")
    assert out["education_requirement"] == "masters"


def test_education_phd_keyword() -> None:
    out = analyze_jd("PhD in Statistics required.")
    assert out["education_requirement"] == "phd"


def test_education_none_when_not_mentioned() -> None:
    out = analyze_jd("We want a great engineer with 3 years of experience.")
    assert out["education_requirement"] is None


# ── required vs preferred classification ─────────────────────────────────────


def test_skills_in_required_section_classified_as_required() -> None:
    """A sentence starting with 'Required:' should route skills to required_skills."""
    out = analyze_jd("Required skills: Python and Docker.")
    # Skills extracted are lowercased by the matcher.
    assert "python" in out["required_skills"]
    assert "python" not in out["preferred_skills"]


def test_skills_in_preferred_section_classified_as_preferred() -> None:
    out = analyze_jd("Required: Python. Nice to have: Kubernetes.")
    # Kubernetes appears in a sentence with `nice to have`.
    assert "kubernetes" in out["preferred_skills"]
    assert "kubernetes" not in out["required_skills"]


def test_skill_appearing_in_both_sections_is_required_only() -> None:
    """De-dup logic: a skill listed in both sections shouldn't double-count."""
    out = analyze_jd("Required: Python. Preferred: Python and Docker.")
    assert out["required_skills"].count("python") == 1
    assert "python" not in out["preferred_skills"]
