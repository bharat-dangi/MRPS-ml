import re
from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from spacy.matcher import PhraseMatcher

from src.ner.taxonomy import load_skill_taxonomy

_EDUCATION_MAP = {
    "phd": "phd", "doctorate": "phd", "master": "masters", "msc": "masters",
    "mba": "masters", "bachelor": "bachelors", "bsc": "bachelors", "b.s.": "bachelors",
    "b.e.": "bachelors", "associate": "associate", "diploma": "diploma",
}

# "N years" or "N+ years". The negative-lookahead on "old" prevents the
# common false positive of capturing age ("I'm 20 years old") as experience.
# Other non-experience uses of "N years" (e.g. "5 years ago") are accepted
# as a trade-off for matching loose resume phrasings like "Experience: 5 years".
_EXP_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\b(?!\s+old)",
    re.IGNORECASE,
)
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
    """Load spaCy model with an EntityRuler (~5k O*NET/ESCO-style SKILL
    patterns) plus a fallback PhraseMatcher for case-insensitive exact match."""
    import warnings
    nlp = spacy.load("en_core_web_sm")
    skills = load_skill_taxonomy()

    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"phrase_matcher_attr": "LOWER", "validate": True, "overwrite_ents": False},
    )
    # W012 fires because entity_ruler internally re-parses patterns; harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="\\[W012\\]")
        ruler.add_patterns([{"label": "SKILL", "pattern": s} for s in skills])

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILL", list(nlp.tokenizer.pipe(skills)))
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

    # Skills via PhraseMatcher. When a shorter match overlaps with a longer one
    # (e.g. "ifrs" inside "ifrs accounting"), keep only the longer form so the
    # UI doesn't show both noisy variants.
    raw_matches = sorted(matcher(doc), key=lambda m: (m[1], -(m[2] - m[1])))
    accepted: list[tuple[int, int]] = []
    for _, start, end in raw_matches:
        if any(a_start <= start and end <= a_end for a_start, a_end in accepted):
            continue
        accepted = [(a_s, a_e) for a_s, a_e in accepted if not (start <= a_s and a_e <= end)]
        accepted.append((start, end))
    skills = list(dict.fromkeys(doc[s:e].text.lower() for s, e in accepted))
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
