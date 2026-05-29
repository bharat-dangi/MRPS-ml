"""Coverage for fairness.py — compute_accent_parity and run_parity_check."""
import logging

import pytest

from src.fairness import (
    PARITY_THRESHOLD_TEXT,
    PARITY_THRESHOLD_VIDEO,
    compute_accent_parity,
    compute_demographic_parity,
    run_parity_check,
)

# ── compute_demographic_parity edge cases ────────────────────────────────────


def test_demographic_parity_empty_groups_returns_pass() -> None:
    """No data → no measurable bias → passes by default."""
    result = compute_demographic_parity([], [])
    assert result["gap"] == pytest.approx(0.0)
    assert result["passes"] is True


def test_demographic_parity_one_empty_group_returns_pass() -> None:
    result = compute_demographic_parity([0.5, 0.6], [])
    assert result["passes"] is True


def test_demographic_parity_below_threshold_passes() -> None:
    result = compute_demographic_parity([0.700, 0.710], [0.705, 0.715])
    assert result["gap"] <= PARITY_THRESHOLD_TEXT
    assert result["passes"] is True


def test_demographic_parity_above_threshold_fails() -> None:
    result = compute_demographic_parity([0.90, 0.92], [0.60, 0.62])
    assert result["gap"] > PARITY_THRESHOLD_TEXT
    assert result["passes"] is False


# ── compute_accent_parity ────────────────────────────────────────────────────


def test_accent_parity_empty_groups_returns_pass() -> None:
    result = compute_accent_parity([], [0.8])
    assert result["passes"] is True
    assert result["gap"] == pytest.approx(0.0)


def test_accent_parity_below_threshold_passes() -> None:
    """Video parity threshold is 3% — a 1.5% gap passes."""
    result = compute_accent_parity([0.80, 0.82], [0.795, 0.805])
    assert result["gap"] <= PARITY_THRESHOLD_VIDEO
    assert result["passes"] is True


def test_accent_parity_above_threshold_fails() -> None:
    result = compute_accent_parity([0.85, 0.86], [0.55, 0.56])
    assert result["gap"] > PARITY_THRESHOLD_VIDEO
    assert result["passes"] is False


def test_accent_parity_returns_all_expected_keys() -> None:
    result = compute_accent_parity([0.8], [0.7])
    assert set(result) == {"mean_native", "mean_non_native", "gap", "passes"}


# ── run_parity_check ─────────────────────────────────────────────────────────


def test_run_parity_check_passes_within_text_threshold(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        result = run_parity_check([0.7, 0.71], [0.705, 0.715], label="text")
    assert result is True
    assert any("PASS" in r.message for r in caplog.records)


def test_run_parity_check_fails_above_threshold(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = run_parity_check([0.90, 0.91], [0.60, 0.61], label="text")
    assert result is False
    assert any("FAIL" in r.message for r in caplog.records)


def test_run_parity_check_label_only_affects_log_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`run_parity_check` delegates to compute_demographic_parity for the
    pass/fail decision, which is *always* evaluated against the TEXT threshold.
    The `label` argument only switches the threshold printed in the log line —
    it does NOT change the decision. Callers needing the video threshold should
    call compute_accent_parity directly.
    """
    a = [0.80, 0.81]
    b = [0.77, 0.78]  # gap ≈ 3% — above text threshold (2%), at video threshold (3%)

    # Both calls return False because the underlying decision uses the text threshold.
    assert run_parity_check(a, b, label="text") is False
    assert run_parity_check(a, b, label="video") is False

    # But the log line for label="video" mentions the 0.03 threshold.
    with caplog.at_level(logging.WARNING):
        run_parity_check(a, b, label="video")
    assert any("threshold=0.03" in r.message for r in caplog.records)
