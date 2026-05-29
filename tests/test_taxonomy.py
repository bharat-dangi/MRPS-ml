"""Skill taxonomy coverage and integrity tests."""

from src.ner.taxonomy import load_skill_taxonomy, taxonomy_domains


def test_taxonomy_has_at_least_5000_unique_skills():
    skills = load_skill_taxonomy()
    assert len({s.lower() for s in skills}) >= 5_000, (
        f"Taxonomy has {len({s.lower() for s in skills})} unique skills, "
        "below the CSV target of 5,000."
    )


def test_taxonomy_spans_required_domains():
    domains = taxonomy_domains()
    # CSV spec calls out IT, healthcare, finance, marketing, and logistics
    required = {"it_software", "finance", "healthcare", "marketing", "logistics_operations"}
    missing = required - set(domains)
    assert not missing, f"Missing required domain files: {missing}"


def test_csv_specified_check_terms_are_present():
    skills_lower = {s.lower() for s in load_skill_taxonomy()}
    # The CSV explicitly says these terms must be tagged as SKILL.
    for term in ("kubernetes", "phlebotomy", "ifrs accounting", "seo optimisation"):
        assert term in skills_lower, f"Required CSV check term missing: {term!r}"


def test_no_empty_or_whitespace_skills():
    skills = load_skill_taxonomy()
    for s in skills:
        assert s.strip() == s, f"Skill has surrounding whitespace: {s!r}"
        assert s, "Empty skill in taxonomy"


def test_skills_are_reasonable_length():
    """Catch accidental garbage entries — no skill should exceed 80 chars."""
    skills = load_skill_taxonomy()
    long_skills = [s for s in skills if len(s) > 80]
    assert not long_skills, f"Suspiciously long skill entries: {long_skills[:3]}"


def test_skills_have_no_newlines():
    skills = load_skill_taxonomy()
    bad = [s for s in skills if "\n" in s or "\r" in s]
    assert not bad, f"Skill entries with embedded newlines: {bad[:3]}"
