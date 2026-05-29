"""
Evaluation harness for the composite scorer.

Builds a deterministic eval set spanning 5 industries × 20 candidates per JD,
with a hand-coded ground-truth ranking (4 strong + 12 partial + 4 weak per JD).
Reports NDCG@5, NDCG@10, MAP, and Spearman rank correlation under two
configurations:

    A. text-only      (composite formula with `video_score=None`)
    B. text+video     (composite formula with `video_score` for video candidates)

Per-domain and overall numbers, with 95% bootstrap confidence intervals.

Run with:
    python scripts/run_eval.py
Output (terminal + ml/eval/eval_results.json):
    {domain: {metric: {value, ci_low, ci_high}}}
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr

from src.scoring.composite import (
    compute_composite_score,
    compute_education_score,
    compute_experience_score,
    compute_skill_overlap,
)

ML_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ML_ROOT / "eval"
OUT_DIR.mkdir(exist_ok=True)
EVAL_JSON = OUT_DIR / "eval_results.json"
SEED = 42

# ── Eval-set JD corpus (2 JDs per industry, mirrors the EPIC-07 spec) ─────────
JDS = [
    # ── Technology ────────────────────────────────────────────────────────────
    dict(domain="Technology", title="Senior Backend Engineer",
         required=["Python", "FastAPI", "PostgreSQL", "Docker", "REST API"],
         preferred=["Kubernetes", "AWS"],
         min_years=5, edu="bachelors"),
    dict(domain="Technology", title="Machine Learning Engineer",
         required=["Python", "PyTorch", "machine learning", "MLOps", "AWS"],
         preferred=["Kubernetes", "Spark"],
         min_years=4, edu="masters"),
    # ── Finance ───────────────────────────────────────────────────────────────
    dict(domain="Finance", title="Financial Analyst",
         required=["financial analysis", "financial modeling", "budgeting", "variance analysis", "Microsoft Excel"],
         preferred=["Power BI", "SQL"],
         min_years=3, edu="bachelors"),
    dict(domain="Finance", title="Senior Accountant",
         required=["GAAP", "IFRS accounting", "bookkeeping", "tax preparation", "month-end close"],
         preferred=["NetSuite", "Xero"],
         min_years=5, edu="bachelors"),
    # ── Healthcare ────────────────────────────────────────────────────────────
    dict(domain="Healthcare", title="Registered Nurse",
         required=["patient care", "clinical assessment", "IV therapy", "electronic health records", "medication administration"],
         preferred=["ACLS certification", "triage"],
         min_years=2, edu="bachelors"),
    dict(domain="Healthcare", title="Clinical Coordinator",
         required=["patient care", "care coordination", "discharge planning", "clinical guidelines", "evidence-based practice"],
         preferred=["EHR", "Epic"],
         min_years=4, edu="bachelors"),
    # ── Marketing ─────────────────────────────────────────────────────────────
    dict(domain="Marketing", title="Digital Marketing Manager",
         required=["SEO", "Google Analytics", "content marketing", "social media marketing", "campaign management"],
         preferred=["HubSpot", "A/B testing"],
         min_years=4, edu="bachelors"),
    dict(domain="Marketing", title="SEO Specialist",
         required=["SEO", "SEO optimisation", "keyword research", "link building", "Google Search Console"],
         preferred=["SEMrush", "Ahrefs"],
         min_years=3, edu="bachelors"),
    # ── Logistics ─────────────────────────────────────────────────────────────
    dict(domain="Logistics", title="Operations Manager",
         required=["supply chain management", "inventory management", "process optimization", "lean six sigma", "ERP"],
         preferred=["SAP", "Kaizen"],
         min_years=5, edu="bachelors"),
    dict(domain="Logistics", title="Supply Chain Analyst",
         required=["supply chain planning", "demand forecasting", "inventory planning", "spend analysis", "Microsoft Excel"],
         preferred=["Power BI", "SAP"],
         min_years=3, edu="bachelors"),
]

EDU_RANK = {None: 0, "diploma": 1, "associate": 2, "bachelors": 3, "masters": 4, "phd": 5}
EDU_LEVELS = ["diploma", "associate", "bachelors", "masters", "phd"]


def _build_eval_set(rng: random.Random) -> list[dict]:
    """For each JD, build 20 candidates: 4 strong, 12 partial, 4 weak.
    Each candidate gets a `gt_rank` (1..20) reflecting the intended order so
    NDCG can be computed against the ground truth."""
    eval_set: list[dict] = []
    for jd_id, jd in enumerate(JDS):
        candidates: list[dict] = []

        # 4 strong: meets all required + most preferred, exceeds experience and education
        for tier_idx in range(4):
            skills = list(jd["required"])
            skills += rng.sample(jd["preferred"], min(2, len(jd["preferred"])))
            skills += ["communication", "stakeholder management"]
            years = jd["min_years"] + rng.randint(0, 3)
            edu = jd["edu"] if rng.random() < 0.6 else EDU_LEVELS[min(4, EDU_RANK[jd["edu"]] + 1)]
            candidates.append(_make_candidate("strong", tier_idx, skills, years, edu, jd))

        # 12 partial: meets some required + experience ~target
        partial_pool = list(jd["required"]) + list(jd["preferred"]) + [
            "Python", "Microsoft Excel", "communication", "SQL", "Agile",
        ]
        for tier_idx in range(12):
            n_required = rng.randint(max(1, len(jd["required"]) // 2), len(jd["required"]) - 1)
            skills = rng.sample(jd["required"], n_required) + rng.sample(partial_pool, 3)
            years = max(0, jd["min_years"] + rng.randint(-2, 1))
            edu = jd["edu"] if rng.random() < 0.5 else EDU_LEVELS[max(0, EDU_RANK[jd["edu"]] - 1)]
            candidates.append(_make_candidate("partial", tier_idx, skills, years, edu, jd))

        # 4 weak: few required skills, low experience, lower education
        for tier_idx in range(4):
            skills = rng.sample(jd["required"], 1) + ["communication", "teamwork", "problem solving"]
            years = max(0, jd["min_years"] - rng.randint(2, 4))
            edu = EDU_LEVELS[max(0, EDU_RANK[jd["edu"]] - 2)]
            candidates.append(_make_candidate("weak", tier_idx, skills, years, edu, jd))

        # Ground-truth ranks: strong > partial > weak; within each tier, deterministic by tier_idx
        candidates.sort(key=lambda c: ({"strong": 0, "partial": 1, "weak": 2}[c["tier"]], c["tier_idx"]))
        for rank, c in enumerate(candidates, start=1):
            c["gt_rank"] = rank
            c["jd_id"] = jd_id
            c["domain"] = jd["domain"]
            c["jd_title"] = jd["title"]

        # 3 of every 20 candidates have a video resume (~15%)
        video_indices = set(rng.sample(range(20), 3))
        for i, c in enumerate(candidates):
            c["has_video"] = i in video_indices

        eval_set.extend(candidates)
    return eval_set


def _make_candidate(tier: str, tier_idx: int, skills: list[str], years: int, edu: str, jd: dict) -> dict:
    return {
        "tier": tier,
        "tier_idx": tier_idx,
        "skills": list(dict.fromkeys(skills)),
        "years": years,
        "edu": edu,
        "jd_required": jd["required"],
        "jd_preferred": jd["preferred"],
        "min_years": jd["min_years"],
        "min_edu": jd["edu"],
    }


def _semantic_proxy(c: dict, rng: random.Random) -> float:
    """Deterministic semantic proxy: 0.5 + skill_overlap × 0.4 + noise, clipped."""
    overlap = compute_skill_overlap(c["skills"], c["jd_required"])
    base = 0.40 + 0.45 * overlap
    noise = rng.gauss(0, 0.06)
    return max(0.0, min(1.0, base + noise))


def _video_score(c: dict, rng: random.Random) -> float:
    """Deterministic video sub-score: high for strong candidates, lower for weak."""
    base = {"strong": 0.82, "partial": 0.62, "weak": 0.42}[c["tier"]]
    return max(0.0, min(1.0, base + rng.gauss(0, 0.06)))


def _score(c: dict, rng: random.Random, use_video: bool) -> float:
    sem = _semantic_proxy(c, rng)
    sk = compute_skill_overlap(c["skills"], c["jd_required"]) + 0.25 * compute_skill_overlap(c["skills"], c["jd_preferred"])
    sk = min(1.0, sk)
    exp = compute_experience_score(c["years"], c["min_years"])
    edu = compute_education_score(c["edu"], c["min_edu"])
    vid = _video_score(c, rng) if (use_video and c["has_video"]) else None
    return compute_composite_score(sem, sk, exp, edu, vid)


# ── Metric implementations ────────────────────────────────────────────────────
def ndcg_at_k(predicted_order: list[int], gt_order: list[int], k: int) -> float:
    """NDCG@k where relevance = inverse-rank (1 / log2(1+rank))."""
    if k <= 0 or not predicted_order:
        return 0.0
    relevance = {c_id: 1.0 / np.log2(2 + gt_rank - 1) for gt_rank, c_id in enumerate(gt_order, start=1)}
    dcg = 0.0
    for i, c_id in enumerate(predicted_order[:k]):
        rel = relevance.get(c_id, 0.0)
        dcg += rel / np.log2(2 + i)
    ideal = sum(
        relevance[gt_order[i]] / np.log2(2 + i)
        for i in range(min(k, len(gt_order)))
    )
    return float(dcg / ideal) if ideal else 0.0


def mean_average_precision(predicted_order: list[int], gt_order: list[int], cutoff_top_n: int = 8) -> float:
    """MAP where 'relevant' = top-N of ground truth."""
    relevant = set(gt_order[:cutoff_top_n])
    hits = 0
    precisions: list[float] = []
    for i, c_id in enumerate(predicted_order, start=1):
        if c_id in relevant:
            hits += 1
            precisions.append(hits / i)
    return float(mean(precisions)) if precisions else 0.0


def spearman(predicted_order: list[int], gt_order: list[int]) -> float:
    pred_rank = {c_id: rank for rank, c_id in enumerate(predicted_order, start=1)}
    gt_rank = {c_id: rank for rank, c_id in enumerate(gt_order, start=1)}
    ids = sorted(pred_rank)
    pred = [pred_rank[c] for c in ids]
    gt = [gt_rank[c] for c in ids]
    rho, _ = spearmanr(pred, gt)
    return float(rho) if not np.isnan(rho) else 0.0


def _bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(SEED)
    arr = np.array(values)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_resamples)]
    boots = sorted(boots)
    lo = boots[int(alpha / 2 * n_resamples)]
    hi = boots[int((1 - alpha / 2) * n_resamples)]
    return float(lo), float(hi)


# ── Run ───────────────────────────────────────────────────────────────────────
def run() -> dict:
    rng_eval = random.Random(SEED)
    eval_set = _build_eval_set(rng_eval)
    print(f"Eval set: {len(JDS)} JDs × 20 candidates = {len(eval_set)} (cand, jd) pairs")
    print(f"Domains: {sorted({jd['domain'] for jd in JDS})}\n")

    # Score each (jd, candidate). Use a *deterministic* RNG per scoring config so
    # the only difference between text-only and text+video is the formula.
    by_jd: dict[int, list[dict]] = {}
    for c in eval_set:
        by_jd.setdefault(c["jd_id"], []).append(c)

    for jd_id, candidates in by_jd.items():
        # Re-seeded per JD so per-candidate noise is reproducible
        for c_idx, c in enumerate(candidates):
            rng_text = random.Random(SEED * 31 + jd_id * 13 + c_idx)
            rng_video = random.Random(SEED * 31 + jd_id * 13 + c_idx)
            c["score_text_only"] = _score(c, rng_text, use_video=False)
            c["score_text_video"] = _score(c, rng_video, use_video=True)
            c["c_id"] = c_idx

    metrics = ["NDCG@5", "NDCG@10", "MAP", "Spearman"]
    results: dict[str, dict[str, dict[str, dict]]] = {
        "text_only": {},
        "text_video": {},
        "_delta": {},
    }

    per_domain_runs: dict[str, dict[str, list[float]]] = {}

    for jd_id, candidates in by_jd.items():
        domain = candidates[0]["domain"]
        gt_order = [c["c_id"] for c in sorted(candidates, key=lambda x: x["gt_rank"])]
        for config in ("text_only", "text_video"):
            score_key = f"score_{config}"
            pred_order = [c["c_id"] for c in sorted(candidates, key=lambda x: -x[score_key])]
            metrics_for_this_jd = {
                "NDCG@5": ndcg_at_k(pred_order, gt_order, 5),
                "NDCG@10": ndcg_at_k(pred_order, gt_order, 10),
                "MAP": mean_average_precision(pred_order, gt_order, cutoff_top_n=8),
                "Spearman": spearman(pred_order, gt_order),
            }
            per_domain_runs.setdefault(config, {}).setdefault(domain, [])
            per_domain_runs.setdefault(config, {}).setdefault("__overall__", [])
            per_domain_runs[config][domain].append(metrics_for_this_jd)
            per_domain_runs[config]["__overall__"].append(metrics_for_this_jd)

    # Aggregate, bootstrap-CI, write results
    output_doc: dict[str, dict] = {"text_only": {}, "text_video": {}, "delta": {}}
    for config in ("text_only", "text_video"):
        for domain, runs in per_domain_runs[config].items():
            d_out: dict[str, dict] = {}
            for m in metrics:
                values = [r[m] for r in runs]
                lo, hi = _bootstrap_ci(values)
                d_out[m] = {"mean": float(np.mean(values)), "ci_low": lo, "ci_high": hi, "n": len(values)}
            output_doc[config][domain] = d_out

    # Delta = text_video - text_only per domain per metric
    for domain in output_doc["text_only"]:
        delta: dict[str, dict] = {}
        for m in metrics:
            d_text = output_doc["text_only"][domain][m]["mean"]
            d_video = output_doc["text_video"][domain][m]["mean"]
            delta[m] = {"text_only": d_text, "text_video": d_video, "delta": d_video - d_text}
        output_doc["delta"][domain] = delta

    print(f"{'Config':<12} {'Domain':<14} {'NDCG@5':>10} {'NDCG@10':>10} {'MAP':>8} {'Spearman':>10}")
    for config in ("text_only", "text_video"):
        for domain, d in output_doc[config].items():
            domain_disp = "Overall" if domain == "__overall__" else domain
            print(f"{config:<12} {domain_disp:<14} "
                  f"{d['NDCG@5']['mean']:>10.3f} {d['NDCG@10']['mean']:>10.3f} "
                  f"{d['MAP']['mean']:>8.3f} {d['Spearman']['mean']:>10.3f}")
        print()

    with open(EVAL_JSON, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"Saved → {EVAL_JSON.relative_to(Path.cwd())}")
    return output_doc


if __name__ == "__main__":
    run()
