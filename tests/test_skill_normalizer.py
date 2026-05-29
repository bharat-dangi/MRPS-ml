"""Skill aliasing + hierarchy matching.

These lock the behaviour that powers compute_skill_overlap. The matching
must be:
  1. Robust to surface variants (postgres ↔ postgresql, k8s ↔ kubernetes).
  2. Asymmetric on hierarchy — specific implies broader, NOT the reverse.
"""
import pytest

from src.ner.skill_normalizer import (
    ALIASES,
    HIERARCHY,
    canonical,
    expand_candidate,
    normalize,
)
from src.scoring.composite import compute_skill_overlap

# ── canonical / normalize ────────────────────────────────────────────────────


@pytest.mark.parametrize("inp, expected", [
    ("Postgres", "postgresql"),
    ("PSQL", "postgresql"),
    ("K8s", "kubernetes"),
    ("kube", "kubernetes"),
    ("JS", "javascript"),
    ("TS", "typescript"),
    ("aws", "amazon web services"),
    ("gcp", "google cloud platform"),
    ("Node", "node.js"),
    ("NodeJS", "node.js"),
    ("Next", "next.js"),
    ("scikit learn", "scikit-learn"),
    ("sklearn", "scikit-learn"),
    ("c sharp", "c#"),
    ("Cpp", "c++"),
    ("Dot Net", ".net"),
    ("CI/CD", "cicd"),
])
def test_canonical_collapses_known_variants(inp: str, expected: str) -> None:
    assert canonical(inp) == expected


def test_canonical_passes_through_unknown_skills_lowercased() -> None:
    """A skill that has no alias entry comes back lowercased verbatim."""
    assert canonical("SomeNewToolXYZ") == "somenewtoolxyz"


def test_normalize_drops_empties_and_whitespace() -> None:
    assert normalize(["Python", "", "  ", "JS"]) == {"python", "javascript"}


# ── expand_candidate (hierarchy) ─────────────────────────────────────────────


def test_expand_candidate_postgres_adds_sql() -> None:
    """A PostgreSQL practitioner also counts as knowing SQL."""
    expanded = expand_candidate(["PostgreSQL"])
    assert "postgresql" in expanded
    assert "sql" in expanded
    assert "relational database" in expanded


def test_expand_candidate_react_adds_javascript() -> None:
    assert "javascript" in expand_candidate(["React"])


def test_expand_candidate_django_adds_python() -> None:
    assert "python" in expand_candidate(["Django"])


def test_expand_candidate_does_not_recurse_beyond_one_level() -> None:
    """Curated; we explicitly don't follow chains further. Next.js → React,
    but expansion stops there (we don't then expand React → JavaScript twice).
    Verify the direct children appear; deeper nodes do too because they're
    listed directly on Next.js."""
    expanded = expand_candidate(["Next.js"])
    assert "react" in expanded
    assert "javascript" in expanded  # listed directly on Next.js


def test_expand_candidate_no_op_for_unknown_skill() -> None:
    """A skill with no hierarchy entry shouldn't be expanded — just normalised."""
    assert expand_candidate(["custom-internal-tool"]) == {"custom-internal-tool"}


# ── compute_skill_overlap with normalization ─────────────────────────────────


def test_overlap_postgres_candidate_matches_sql_jd() -> None:
    """The motivating case: candidate has PostgreSQL, JD needs SQL → full match."""
    assert compute_skill_overlap(["PostgreSQL"], ["SQL"]) == pytest.approx(1.0)


def test_overlap_alias_collapse_both_directions() -> None:
    """`postgres` (candidate) and `postgresql` (JD) are the same skill."""
    assert compute_skill_overlap(["postgres"], ["PostgreSQL"]) == pytest.approx(1.0)
    assert compute_skill_overlap(["PostgreSQL"], ["postgres"]) == pytest.approx(1.0)


def test_overlap_k8s_alias_matches_kubernetes() -> None:
    assert compute_skill_overlap(["K8s"], ["Kubernetes"]) == pytest.approx(1.0)


def test_overlap_js_alias_matches_javascript() -> None:
    assert compute_skill_overlap(["JS"], ["JavaScript"]) == pytest.approx(1.0)


def test_overlap_react_candidate_matches_javascript_jd() -> None:
    """React implies JavaScript via hierarchy."""
    assert compute_skill_overlap(["React"], ["JavaScript"]) == pytest.approx(1.0)


def test_overlap_django_candidate_matches_python_jd() -> None:
    assert compute_skill_overlap(["Django"], ["Python"]) == pytest.approx(1.0)


def test_overlap_generic_to_specific_gets_partial_credit() -> None:
    """Reverse direction: candidate has generic SQL, JD wants PostgreSQL.
    SQL is the broader parent of PostgreSQL → partial credit (0.5).
    Recruiter intuition: knowing SQL is a transferable foundation."""
    assert compute_skill_overlap(["SQL"], ["PostgreSQL"]) == pytest.approx(0.5)


def test_overlap_python_to_pandas_gets_partial_credit() -> None:
    """Pandas lists Python as a parent in HIERARCHY → partial credit. A
    Python dev has the language foundation pandas needs, but doesn't
    automatically know the API. 0.5 reflects 'transferable but not full'."""
    assert compute_skill_overlap(["Python"], ["Pandas"]) == pytest.approx(0.5)


def test_overlap_sibling_databases_partial_match() -> None:
    """MariaDB and PostgreSQL are both in the rdbms family → 0.5 sibling credit."""
    assert compute_skill_overlap(["MariaDB"], ["PostgreSQL"]) == pytest.approx(0.5)
    assert compute_skill_overlap(["MySQL"], ["PostgreSQL"]) == pytest.approx(0.5)


def test_overlap_sibling_clouds_partial_match() -> None:
    """AWS / GCP / Azure transfer to one another at 0.5."""
    assert compute_skill_overlap(["AWS"], ["Microsoft Azure"]) == pytest.approx(0.5)


def test_overlap_exact_beats_sibling() -> None:
    """If candidate has the exact required skill AND a sibling, exact wins (1.0)."""
    assert compute_skill_overlap(
        ["PostgreSQL", "MySQL"], ["PostgreSQL"],
    ) == pytest.approx(1.0)


def test_overlap_unrelated_still_zero() -> None:
    """Sanity: completely unrelated skills get nothing."""
    assert compute_skill_overlap(["photography"], ["python"]) == pytest.approx(0.0)
    assert compute_skill_overlap(["mariadb"], ["docker"]) == pytest.approx(0.0)


def test_overlap_partial_match_with_hierarchy() -> None:
    """JD requires 3 skills; candidate has PostgreSQL which covers 'sql'.
    They also have React which covers 'javascript'. Score = 2/3."""
    assert compute_skill_overlap(
        ["PostgreSQL", "React"],
        ["SQL", "JavaScript", "Docker"],
    ) == pytest.approx(2 / 3)


def test_overlap_empty_required_still_zero() -> None:
    """Pre-existing contract: no required skills → 0.0, not 1.0."""
    assert compute_skill_overlap(["Python"], []) == pytest.approx(0.0)


def test_overlap_real_world_case_from_user_report() -> None:
    """The exact scenario the user hit: 'candidate have sql skill and job
    need postgres, match not happening'.

    All three of these directions now produce non-zero scores:
      - candidate=MariaDB, JD=SQL: 1.0 (specific → broader via HIERARCHY)
      - candidate=SQL, JD=PostgreSQL: 0.5 (generic broader-than-specific)
      - candidate=MariaDB, JD=PostgreSQL: 0.5 (sibling via FAMILIES)
    """
    assert compute_skill_overlap(["MariaDB"], ["SQL"]) == pytest.approx(1.0)
    assert compute_skill_overlap(["SQL"], ["PostgreSQL"]) == pytest.approx(0.5)
    assert compute_skill_overlap(["MariaDB"], ["PostgreSQL"]) == pytest.approx(0.5)


# ── ALIASES / HIERARCHY data integrity ───────────────────────────────────────


def test_alias_keys_and_values_are_lowercase() -> None:
    for k, v in ALIASES.items():
        assert k == k.lower(), f"alias key {k!r} not lowercase"
        assert v == v.lower(), f"alias value {v!r} not lowercase"


def test_hierarchy_keys_and_values_are_lowercase() -> None:
    for k, parents in HIERARCHY.items():
        assert k == k.lower()
        for p in parents:
            assert p == p.lower()


def test_alias_values_resolve_to_themselves() -> None:
    """A canonical name should be a fixed point of the alias map — otherwise
    we'd have an infinite chain like postgres → psql → postgresql."""
    for canonical_name in ALIASES.values():
        assert ALIASES.get(canonical_name, canonical_name) == canonical_name
