"""
Fairness metrics: demographic parity and accent parity.
These are run after screening to detect bias in the scoring pipeline.
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

PARITY_THRESHOLD_TEXT = 0.02   # ±2% for text resumes (spec requirement)
PARITY_THRESHOLD_VIDEO = 0.03  # ±3% for video resumes (spec requirement)


def compute_demographic_parity(
    group_a_scores: Sequence[float],
    group_b_scores: Sequence[float],
) -> dict[str, float]:
    """Return mean scores and absolute gap between two demographic groups."""
    if not group_a_scores or not group_b_scores:
        return {"mean_a": 0.0, "mean_b": 0.0, "gap": 0.0, "passes": True}
    mean_a = sum(group_a_scores) / len(group_a_scores)
    mean_b = sum(group_b_scores) / len(group_b_scores)
    gap = abs(mean_a - mean_b)
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "gap": gap,
        "passes": gap <= PARITY_THRESHOLD_TEXT,
    }


def compute_accent_parity(
    native_scores: Sequence[float],
    non_native_scores: Sequence[float],
) -> dict[str, float]:
    """Return parity metrics for accented speech in video scoring."""
    if not native_scores or not non_native_scores:
        return {"mean_native": 0.0, "mean_non_native": 0.0, "gap": 0.0, "passes": True}
    mean_native = sum(native_scores) / len(native_scores)
    mean_non_native = sum(non_native_scores) / len(non_native_scores)
    gap = abs(mean_native - mean_non_native)
    return {
        "mean_native": mean_native,
        "mean_non_native": mean_non_native,
        "gap": gap,
        "passes": gap <= PARITY_THRESHOLD_VIDEO,
    }


def run_parity_check(
    group_a_scores: Sequence[float],
    group_b_scores: Sequence[float],
    label: str = "text",
) -> bool:
    """Log parity results and return True if the check passes."""
    threshold = PARITY_THRESHOLD_VIDEO if label == "video" else PARITY_THRESHOLD_TEXT
    metrics = compute_demographic_parity(group_a_scores, group_b_scores)
    if metrics["passes"]:
        logger.info("[Fairness/%s] PASS — gap=%.4f (threshold=%.2f)", label, metrics["gap"], threshold)
    else:
        logger.warning("[Fairness/%s] FAIL — gap=%.4f exceeds threshold=%.2f", label, metrics["gap"], threshold)
    return metrics["passes"]
