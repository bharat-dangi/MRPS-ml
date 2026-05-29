"""
Section classifier tests.

These exercise the regex fallback path so they can pass in CI without spaCy /
the trained model. The trained-model branch is covered indirectly by
scripts/train_section_classifier.py (which exits non-zero if val accuracy
drops below 0.92).
"""
import pytest

from src.ner.section_classifier import _regex_classify, classify_section, segment_resume


@pytest.mark.parametrize(
    "text, expected",
    [
        ("EDUCATION\nBachelor of Computer Science", "Education"),
        ("Work Experience\nSenior Engineer at Acme Corp", "Experience"),
        ("Professional Experience\n2020-2025 Acme", "Experience"),
        ("Technical Skills\nPython, FastAPI, Docker", "Skills"),
        ("Skills\nPython", "Skills"),
        ("Summary\nDetail-oriented engineer", "Summary"),
        ("Career Summary\nResults-driven specialist", "Summary"),
        ("References\nAvailable on request", "Other"),
        ("Hobbies\nbushwalking, photography", "Other"),
    ],
)
def test_regex_classify_recognises_common_headers(text, expected):
    label, conf = _regex_classify(text)
    assert label == expected
    assert conf == 1.0


def test_regex_classify_returns_other_for_unknown_header():
    label, conf = _regex_classify("Some random unstructured text without a header.")
    assert label == "Other"
    assert conf == 0.0


def test_classify_section_handles_empty_input():
    label, conf, source = classify_section("")
    assert label == "Other"
    assert conf == 0.0
    assert source == "regex_fallback"


def test_segment_resume_picks_up_multiple_sections():
    resume = (
        "Jane Doe\n"
        "SUMMARY\nDetail-oriented engineer with 7+ years of experience.\n"
        "WORK EXPERIENCE\nSenior Backend Engineer | Acme Corp | 2020-2025\n"
        "EDUCATION\nBachelor of Computer Science — UNSW\n"
        "SKILLS\nPython, Docker, Kubernetes, PostgreSQL\n"
        "REFERENCES\nAvailable on request."
    )
    sections = segment_resume(resume)
    # At minimum each labelled header section should produce one block
    assert sum(len(blocks) for blocks in sections.values()) >= 5
    # The 5 named sections must each be populated when the trained classifier
    # is available; the regex fallback alone may misclassify the header-less
    # name block as Other / Skills but the named sections are stable.
    assert any("Bachelor" in b for b in sections.get("Education", [])) or any(
        "EDUCATION" in b for b in sections.get("Education", [])
    )
