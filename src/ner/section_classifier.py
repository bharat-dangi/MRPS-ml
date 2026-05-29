"""
Resume section classifier.

Combines a trained spaCy textcat model (when available) with a regex-based
fallback that fires when the classifier confidence drops below 0.70 — or when
the trained model isn't installed yet.

Public API:
    classify_section(text: str) -> tuple[str, float, str]
        Returns (label, confidence, source) where source is "classifier" or
        "regex_fallback".

    segment_resume(text: str) -> dict[str, list[str]]
        Walks a raw resume top-to-bottom, returning a dict of
        {label: [text_blocks]}.

Labels: Education | Experience | Skills | Summary | Other
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import spacy

LABELS = ("Education", "Experience", "Skills", "Summary", "Other")
CONFIDENCE_THRESHOLD = 0.70

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "section_classifier"

# Section header patterns are matched against lowercased line text. Order
# matters within a label: a longer phrase ("work experience") wins over a
# shorter one ("experience") because earlier entries in the same tuple are
# tried first.
_REGEX_HEADERS: dict[str, tuple[str, ...]] = {
    "Education": (
        "education", "academic background", "academic qualifications",
        "qualifications", "academic history", "degrees", "training and education",
    ),
    "Experience": (
        "work experience", "professional experience", "employment history",
        "career history", "work history", "relevant experience", "experience",
    ),
    "Skills": (
        "technical skills", "core competencies", "skills and expertise",
        "key skills", "skills summary", "skills", "competencies",
        "areas of expertise",
    ),
    "Summary": (
        "professional summary", "career summary", "personal summary",
        "summary of qualifications", "summary", "objective", "career objective",
        "profile", "about me",
    ),
    "Other": (
        "references", "hobbies", "interests", "languages",
        "certifications", "publications", "volunteering", "awards",
    ),
}


@lru_cache(maxsize=1)
def _load_classifier():
    """Load the trained spaCy textcat model. Returns None if not present."""
    if not _MODEL_DIR.exists():
        return None
    try:
        return spacy.load(_MODEL_DIR)
    except Exception:
        return None


def _regex_classify(text: str) -> tuple[str, float]:
    """Detect a section header in the first 2 lines. Confidence is binary:
    1.0 if a known header is found, 0.0 otherwise."""
    head = "\n".join(text.lower().split("\n")[:2])
    for label, patterns in _REGEX_HEADERS.items():
        for pat in patterns:
            if re.search(rf"(?:^|\n)\s*{re.escape(pat)}\s*[:\n]?", head):
                return label, 1.0
    return "Other", 0.0


def classify_section(text: str) -> tuple[str, float, str]:
    """Classify a single text block. Falls back to regex when the trained
    classifier is missing or its top score is below `CONFIDENCE_THRESHOLD`."""
    if not text.strip():
        return "Other", 0.0, "regex_fallback"

    nlp = _load_classifier()
    if nlp is not None:
        doc = nlp(text)
        if doc.cats:
            label, conf = max(doc.cats.items(), key=lambda kv: kv[1])
            if conf >= CONFIDENCE_THRESHOLD:
                return label, float(conf), "classifier"

    label, conf = _regex_classify(text)
    return label, conf, "regex_fallback"


# Header lines that signal a section change when they appear alone on a line.
_HEADER_LINE_RE = re.compile(
    r"^\s*(education|work experience|professional experience|employment history|"
    r"experience|technical skills|skills|core competencies|key skills|"
    r"summary|professional summary|career summary|objective|profile|"
    r"references|hobbies|interests|languages|certifications|publications|"
    r"awards|volunteering)\s*[:\-]?\s*$",
    re.IGNORECASE,
)


def segment_resume(text: str) -> dict[str, list[str]]:
    """Split a resume into labelled sections. Section boundaries are detected
    by header lines; the classifier labels each block's contents (header +
    body) and the dict accumulates blocks per label."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if _HEADER_LINE_RE.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    sections: dict[str, list[str]] = {label: [] for label in LABELS}
    for block in blocks:
        if not block:
            continue
        label, _conf, _source = classify_section(block)
        sections[label].append(block)
    return sections
