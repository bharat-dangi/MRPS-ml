import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

_VOCAB_PATH = Path(__file__).parent / "skills_vocab.json"

_EDUCATION_MAP = {
    "phd": "phd", "doctorate": "phd", "master": "masters", "msc": "masters",
    "mba": "masters", "bachelor": "bachelors", "bsc": "bachelors", "b.s.": "bachelors",
    "b.e.": "bachelors", "associate": "associate", "diploma": "diploma",
}

_EXP_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_NAME_LIKE = re.compile(r"^([A-Z][a-z]+ [A-Z][a-z]+)")


@dataclass
class CandidateProfile:
    full_name: str = "Unknown"
    email: str | None = None
    phone: str | None = None
    skills: list[str] = field(default_factory=list)
    skill_sources: dict[str, str] = field(default_factory=dict)  # skill → "text" | "video"
    years_experience: float = 0.0
    education_level: str | None = None
    raw_text: str = ""


@lru_cache(maxsize=1)
def _load_nlp():
    """Load spaCy model and PhraseMatcher once per process."""
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    with open(_VOCAB_PATH) as f:
        skills: list[str] = json.load(f)

    patterns = list(nlp.pipe(skills))
    matcher.add("SKILL", patterns)
    return nlp, matcher


def _extract_education(text: str) -> str | None:
    lower = text.lower()
    for keyword, level in _EDUCATION_MAP.items():
        if keyword in lower:
            return level
    return None


def _extract_experience(text: str) -> float:
    matches = _EXP_PATTERN.findall(text)
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def extract_candidate_profile(raw_text: str) -> CandidateProfile:
    """Run spaCy NER + PhraseMatcher to extract structured candidate data."""
    nlp, matcher = _load_nlp()
    doc = nlp(raw_text[:50_000])  # cap at 50k chars to limit latency

    # Email and phone via regex (more reliable than NER)
    email_match = _EMAIL_PATTERN.search(raw_text)
    phone_match = _PHONE_PATTERN.search(raw_text)

    # Candidate name: first PERSON entity or first line matching Name pattern
    name = "Unknown"
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    if name == "Unknown":
        first_line = raw_text.strip().split("\n")[0]
        m = _NAME_LIKE.match(first_line)
        if m:
            name = m.group(1)

    # Skills via PhraseMatcher — all sourced from resume text
    skill_matches = matcher(doc)
    skills = list(dict.fromkeys(
        doc[start:end].text.lower()
        for _, start, end in skill_matches
    ))
    skill_sources = dict.fromkeys(skills, "text")

    return CandidateProfile(
        full_name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        skills=skills,
        skill_sources=skill_sources,
        years_experience=_extract_experience(raw_text),
        education_level=_extract_education(raw_text),
        raw_text=raw_text,
    )
