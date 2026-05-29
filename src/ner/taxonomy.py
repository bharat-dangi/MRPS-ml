"""
Skill taxonomy loader.

Aggregates the curated skill lists in `ml/data/skills/*.json` plus the legacy
`skills_vocab.json` into a single de-duplicated list keyed by lowercase form.

The combined taxonomy spans IT, finance, healthcare, marketing, logistics,
manufacturing, HR/sales/design, legal/education, plus 300+ certifications and
300+ cross-domain transferable skills (~5,000 unique terms).

Usage:
    from src.ner.taxonomy import load_skill_taxonomy
    skills = load_skill_taxonomy()  # cached after first call
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_TAXONOMY_DIR = Path(__file__).resolve().parents[2] / "data" / "skills"
_LEGACY_VOCAB = Path(__file__).parent / "skills_vocab.json"


@lru_cache(maxsize=1)
def load_skill_taxonomy() -> list[str]:
    """Return a de-duplicated list of skill canonical forms.

    De-duplication is case-insensitive: the first-seen capitalisation wins,
    so 'Python' beats 'python' if both appear. Caller can lowercase as needed.
    """
    seen: dict[str, str] = {}

    if _LEGACY_VOCAB.exists():
        with open(_LEGACY_VOCAB) as f:
            for skill in json.load(f):
                key = skill.lower().strip()
                if key and key not in seen:
                    seen[key] = skill.strip()

    if _TAXONOMY_DIR.exists():
        for path in sorted(_TAXONOMY_DIR.glob("*.json")):
            with open(path) as f:
                for skill in json.load(f):
                    key = skill.lower().strip()
                    if key and key not in seen:
                        seen[key] = skill.strip()

    return list(seen.values())


@lru_cache(maxsize=1)
def taxonomy_domains() -> dict[str, int]:
    """Per-domain skill counts (for /docs eval reporting)."""
    counts: dict[str, int] = {}
    if not _TAXONOMY_DIR.exists():
        return counts
    for path in sorted(_TAXONOMY_DIR.glob("*.json")):
        with open(path) as f:
            counts[path.stem] = len(json.load(f))
    return counts
