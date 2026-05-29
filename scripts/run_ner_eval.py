"""
NER F1 evaluation: precision / recall / F1 per entity type (SKILL, DEGREE,
JOB_TITLE, CERT) against a deterministic test set of 100 resume blocks
spanning the same 5 industries as the ranking eval.

We don't have a hand-annotated test set; instead each block is generated from
templates that carry their own gold entity labels — the resulting numbers are
honest given the synthetic data, and the same harness can run against a real
labelled set by replacing the generator with a JSONL loader.

Output: ml/eval/ner_eval.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.ner.extractor import extract_candidate_profile, _load_nlp  # noqa: F401  (load patterns)

ML_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ML_ROOT / "eval" / "ner_eval.json"
OUT_PATH.parent.mkdir(exist_ok=True)
SEED = 42

DEGREES_GOLD = [
    "Bachelor of Science", "Bachelor of Computer Science", "Master of Information Technology",
    "MBA", "PhD in Machine Learning", "Bachelor of Engineering", "Diploma",
    "Associate's degree",
]
JOB_TITLES_GOLD = [
    "Senior Backend Engineer", "Machine Learning Engineer", "Registered Nurse",
    "Financial Analyst", "Digital Marketing Manager", "Supply Chain Analyst",
    "HR Business Partner", "Operations Manager",
]
CERTS_GOLD = [
    "PMP", "Scrum Master", "CISSP", "AWS Certified Solutions Architect",
    "BLS certification", "ACLS certification", "CFA Level I", "Six Sigma Green Belt",
    "Google Analytics Individual Qualification",
]

# Per-block we sample one or more "gold" entities and inject them into a
# realistic-looking resume snippet. The extractor must return at least the
# injected spans (case-insensitive) to count as a hit.

SKILL_POOLS_FOR_NER = {
    "tech": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "machine learning", "PyTorch", "AWS"],
    "finance": ["financial modeling", "IFRS accounting", "budgeting", "Microsoft Excel", "Power BI", "GAAP"],
    "health": ["patient care", "IV therapy", "phlebotomy", "electronic health records", "Epic"],
    "marketing": ["SEO", "SEO optimisation", "Google Analytics", "content marketing", "campaign management"],
    "logistics": ["supply chain management", "inventory management", "Lean Six Sigma", "ERP", "SAP"],
}


def _generate_block(rng: random.Random) -> dict:
    """Return {text, gold: {SKILL: [...], DEGREE: [...], JOB_TITLE: [...], CERT: [...]}}."""
    domain = rng.choice(list(SKILL_POOLS_FOR_NER.keys()))
    skills = rng.sample(SKILL_POOLS_FOR_NER[domain], k=rng.randint(3, 5))
    degree = rng.choice(DEGREES_GOLD)
    title = rng.choice(JOB_TITLES_GOLD)
    cert = rng.choice(CERTS_GOLD) if rng.random() < 0.5 else None

    cert_line = f"\nCertifications: {cert}." if cert else ""
    text = (
        f"{title} | Acme Group | 2020–2025\n"
        f"Delivered cross-functional initiatives using {', '.join(skills)}.\n"
        f"Education: {degree} — University of Sydney (2018).\n"
        f"Skills: {', '.join(skills)}.{cert_line}"
    )
    gold = {
        "SKILL": [s.lower() for s in skills],
        "DEGREE": [degree.lower()],
        "JOB_TITLE": [title.lower()],
        "CERT": [cert.lower()] if cert else [],
    }
    return {"text": text, "gold": gold}


def _entity_check(pred: set[str], gold: list[str]) -> tuple[int, int, int]:
    """Return (tp, fp, fn) treating each entity by case-insensitive containment."""
    matched = {g for g in gold if g in pred}
    tp = len(matched)
    fn = len(gold) - tp
    fp = len(pred - {g for g in gold})
    return tp, fp, fn


def run(n_samples: int = 100) -> dict:
    rng = random.Random(SEED)
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in ("SKILL", "DEGREE", "JOB_TITLE", "CERT")}

    for _ in range(n_samples):
        block = _generate_block(rng)
        profile = extract_candidate_profile(block["text"])

        # SKILL extraction comes from the extractor
        pred_skills = {s.lower() for s in profile.skills}
        tp, fp, fn = _entity_check(pred_skills, block["gold"]["SKILL"])
        counts["SKILL"]["tp"] += tp
        counts["SKILL"]["fp"] += fp
        counts["SKILL"]["fn"] += fn

        # DEGREE: regex-based normalisation maps to a level. We check that the
        # extractor returned *some* education_level (predicted positive); the
        # gold's degree level is derived from its keyword for level comparison.
        gold_lvl = _gold_to_level(block["gold"]["DEGREE"][0])
        pred_lvl = profile.education_level
        if gold_lvl is not None and pred_lvl is not None:
            counts["DEGREE"]["tp"] += 1 if gold_lvl == pred_lvl else 0
            counts["DEGREE"]["fp"] += 0 if gold_lvl == pred_lvl else 1
            counts["DEGREE"]["fn"] += 0 if gold_lvl == pred_lvl else 1
        elif gold_lvl is None and pred_lvl is None:
            pass  # true negative — not counted
        else:
            # one side empty, the other not
            counts["DEGREE"]["fp" if pred_lvl else "fn"] += 1

        # JOB_TITLE: the extractor doesn't tag JOB_TITLE entities explicitly,
        # but it does pick up "engineer/analyst/etc." inside skills if those
        # phrases appear in the taxonomy. We mark this as recall=0 unless the
        # full title appears verbatim in `skills` (rare). This honestly reflects
        # the current capability and motivates a future fine-tuned NER model.
        title_norm = block["gold"]["JOB_TITLE"][0]
        if any(title_norm in s for s in pred_skills):
            counts["JOB_TITLE"]["tp"] += 1
        else:
            counts["JOB_TITLE"]["fn"] += 1

        # CERT: certifications taxonomy was loaded as SKILL patterns, so a
        # cert hit shows up as a SKILL extraction. Count it both as SKILL (above)
        # and as a CERT hit here for the per-entity breakdown.
        if block["gold"]["CERT"]:
            cert_norm = block["gold"]["CERT"][0]
            if cert_norm in pred_skills:
                counts["CERT"]["tp"] += 1
            else:
                counts["CERT"]["fn"] += 1

    results: dict[str, dict] = {}
    for label, c in counts.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": c["tp"] + c["fn"],
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }

    print(f"{'Entity':<10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}")
    for label, r in results.items():
        print(f"{label:<10} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>8.3f} {r['support']:>10d}")

    with open(OUT_PATH, "w") as f:
        json.dump({"n_samples": n_samples, "per_entity": results}, f, indent=2)
    print(f"\nSaved → {OUT_PATH.relative_to(Path.cwd())}")
    return results


def _gold_to_level(degree_text: str) -> str | None:
    t = degree_text.lower()
    if "phd" in t or "doctor" in t:
        return "phd"
    if "master" in t or "mba" in t or "msc" in t:
        return "masters"
    if "bachelor" in t or "bsc" in t:
        return "bachelors"
    if "associate" in t:
        return "associate"
    if "diploma" in t:
        return "diploma"
    return None


if __name__ == "__main__":
    run()
