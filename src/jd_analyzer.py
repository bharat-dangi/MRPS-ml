import json
import re
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

_VOCAB_PATH = Path(__file__).parent / "ner" / "skills_vocab.json"

_EXP_PATTERN = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)
_EDU_KEYWORDS = {
    "phd": "phd", "doctorate": "phd",
    "master": "masters", "msc": "masters", "mba": "masters",
    "bachelor": "bachelors", "bsc": "bachelors",
    "associate": "associate", "diploma": "diploma",
}

_REQUIRED_MARKERS = re.compile(
    r"required|must have|mandatory|essential|minimum|qualifications",
    re.IGNORECASE,
)
_PREFERRED_MARKERS = re.compile(
    r"preferred|nice to have|bonus|desirable|advantage|plus",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_nlp():
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    with open(_VOCAB_PATH) as f:
        skills: list[str] = json.load(f)
    patterns = list(nlp.pipe(skills))
    matcher.add("SKILL", patterns)
    return nlp, matcher


def analyze_jd(jd_text: str) -> dict:
    """
    Parse a job description into structured fields.
    Returns: {required_skills, preferred_skills, min_experience, education_requirement}
    """
    nlp, matcher = _load_nlp()
    doc = nlp(jd_text[:30_000])

    # Classify required vs preferred by sentence context
    required_skills: list[str] = []
    preferred_skills: list[str] = []

    for sent in doc.sents:
        sent_text = sent.text
        sent_doc = nlp(sent_text)
        sent_skills = [
            sent_doc[s:e].text.lower()
            for _, s, e in matcher(sent_doc)
        ]
        if not sent_skills:
            continue
        if _REQUIRED_MARKERS.search(sent_text):
            required_skills.extend(sent_skills)
        elif _PREFERRED_MARKERS.search(sent_text):
            preferred_skills.extend(sent_skills)
        else:
            required_skills.extend(sent_skills)

    # De-duplicate while preserving order
    required_skills = list(dict.fromkeys(required_skills))
    preferred_skills = list(dict.fromkeys(s for s in preferred_skills if s not in required_skills))

    # Minimum experience
    exp_matches = _EXP_PATTERN.findall(jd_text)
    min_experience = min((int(m) for m in exp_matches), default=0)

    # Education requirement
    education_requirement = None
    lower_jd = jd_text.lower()
    for keyword, level in _EDU_KEYWORDS.items():
        if keyword in lower_jd:
            education_requirement = level
            break

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "min_experience": min_experience,
        "education_requirement": education_requirement,
    }
