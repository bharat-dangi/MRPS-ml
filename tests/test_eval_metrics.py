"""Unit tests for the pure ranking metrics in scripts/run_eval.py.

These power EPIC-07 model evaluation — NDCG, MAP, Spearman, bootstrap CI.
All four are deterministic, side-effect-free, and easy to property-test.
"""
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

import pytest

from run_eval import (  # noqa: E402
    _bootstrap_ci,
    mean_average_precision,
    ndcg_at_k,
    spearman,
)


# ── ndcg_at_k ────────────────────────────────────────────────────────────────


def test_ndcg_perfect_ranking_is_one() -> None:
    gt = [1, 2, 3, 4, 5]
    assert ndcg_at_k(gt, gt, k=5) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_is_less_than_one() -> None:
    gt = [1, 2, 3, 4, 5]
    pred = list(reversed(gt))
    assert ndcg_at_k(pred, gt, k=5) < 1.0


def test_ndcg_irrelevant_predictions_score_zero() -> None:
    """If none of the predicted items appear in ground truth, DCG is 0 → NDCG is 0."""
    assert ndcg_at_k([99, 98, 97], [1, 2, 3], k=3) == 0.0


def test_ndcg_k_zero_returns_zero() -> None:
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=0) == 0.0


def test_ndcg_empty_prediction_returns_zero() -> None:
    assert ndcg_at_k([], [1, 2, 3], k=5) == 0.0


def test_ndcg_swap_top_two_is_below_one_but_high() -> None:
    """Swapping top-2 should still score high but strictly less than 1.0."""
    gt = [1, 2, 3, 4, 5]
    swapped = [2, 1, 3, 4, 5]
    val = ndcg_at_k(swapped, gt, k=5)
    assert 0.9 < val < 1.0


# ── mean_average_precision ───────────────────────────────────────────────────


def test_map_perfect_ranking_with_full_recall_is_one() -> None:
    """If we predict all relevant items in the right order, MAP is 1.0."""
    gt = [1, 2, 3, 4, 5]
    assert mean_average_precision(gt, gt, cutoff_top_n=5) == pytest.approx(1.0)


def test_map_no_hits_is_zero() -> None:
    assert mean_average_precision([99, 98, 97], [1, 2, 3], cutoff_top_n=3) == 0.0


def test_map_hit_at_rank_one_only() -> None:
    """If only the first item is relevant, MAP = 1/1 = 1.0."""
    assert mean_average_precision([1], [1, 2, 3], cutoff_top_n=3) == pytest.approx(1.0)


def test_map_relevant_at_rank_two() -> None:
    """[wrong, right] → P@2 = 1/2 → MAP = 0.5."""
    assert mean_average_precision([99, 1], [1, 2, 3], cutoff_top_n=3) == pytest.approx(0.5)


# ── spearman ─────────────────────────────────────────────────────────────────


def test_spearman_perfect_correlation_is_one() -> None:
    order = [1, 2, 3, 4, 5]
    assert spearman(order, order) == pytest.approx(1.0)


def test_spearman_reversed_is_negative_one() -> None:
    gt = [1, 2, 3, 4, 5]
    assert spearman(list(reversed(gt)), gt) == pytest.approx(-1.0)


def test_spearman_single_pair_returns_zero() -> None:
    """scipy.stats.spearmanr returns NaN for a single pair; we coerce to 0."""
    assert spearman([1], [1]) == 0.0


# ── _bootstrap_ci ────────────────────────────────────────────────────────────


def test_bootstrap_ci_empty_returns_zero_zero() -> None:
    assert _bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_constant_values_collapse_to_value() -> None:
    """If every sample is identical, both CI bounds equal that value."""
    lo, hi = _bootstrap_ci([0.7] * 100, n_resamples=200)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_bootstrap_ci_bracket_contains_mean() -> None:
    """The 95% CI should usually bracket the sample mean."""
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.95]
    lo, hi = _bootstrap_ci(values, n_resamples=500)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
