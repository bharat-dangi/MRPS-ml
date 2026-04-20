import numpy as np

from src.scoring.composite import WEIGHTS


def compute_shap_values(
    feature_scores: dict[str, float],
    baseline_scores: dict[str, float],
) -> dict[str, float]:
    """
    Analytical SHAP for a linear model: shap_i = w_i * (x_i - baseline_i).
    """

    shap_values: dict[str, float] = {}

    for feature in WEIGHTS:
        value = feature_scores.get(feature)

        if value is None:
            value = 0.0

        baseline = baseline_scores.get(feature, 0.5)

        shap_values[feature] = WEIGHTS[feature] * (value - baseline)

    return shap_values


def compute_population_baselines(all_feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean value per feature across all candidates — used as SHAP baseline."""
    if not all_feature_dicts:
        return {k: 0.5 for k in WEIGHTS}

    totals: dict[str, float] = {k: 0.0 for k in WEIGHTS}

    for d in all_feature_dicts:
        for k in WEIGHTS:
            value = d.get(k)
            totals[k] += value if value is not None else 0.0

    n = len(all_feature_dicts)
    return {k: totals[k] / n for k in WEIGHTS}
