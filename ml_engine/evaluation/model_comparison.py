"""Model comparison and ranking layer based EXCLUSIVELY on validation metrics.

Strict Leakage & Evaluation Rules:
1. Candidate model ranking is determined ONLY by VALIDATION metrics.
2. TEST set metrics are included in output for display/reporting but NEVER affect rank or selection.
3. Supported selection metrics: 'rmse', 'mae', 'mape', 'r2'.
4. Metric direction:
   - 'rmse', 'mae', 'mape': lower validation value is better.
   - 'r2': higher validation value is better.
"""

from __future__ import annotations

from typing import Any


class ModelComparisonError(ValueError):
    """Raised when model comparison or ranking fails."""


SUPPORTED_SELECTION_METRICS = ("rmse", "mae", "mape", "r2")


def compare_models(
    reports: list[dict[str, Any]],
    selection_metric: str = "rmse",
) -> dict[str, Any]:
    """Compare a list of model reports and return structured validation-based ranking results.

    Parameters
    ----------
    reports : list[dict[str, Any]]
        List of candidate model output reports containing 'model', 'validation_metrics',
        and optionally 'test_metrics'.
    selection_metric : str
        Metric key used for ranking ('rmse', 'mae', 'mape', 'r2'). Default: 'rmse'.

    Returns
    -------
    dict[str, Any]
        Structured JSON-serializable comparison dictionary containing selection_metric,
        selection_split="validation", ranking list, and recommended_model.
    """
    if not reports:
        raise ModelComparisonError("Reports list cannot be empty")

    metric_key = selection_metric.lower().strip()
    if metric_key not in SUPPORTED_SELECTION_METRICS:
        raise ModelComparisonError(
            f"Unsupported selection metric '{selection_metric}'. "
            f"Supported metrics: {list(SUPPORTED_SELECTION_METRICS)}"
        )

    validated_candidates = []
    for idx, r in enumerate(reports):
        if not isinstance(r, dict):
            raise ModelComparisonError(f"Report at index {idx} must be a dictionary")

        model_name = r.get("model")
        if not model_name:
            raise ModelComparisonError(f"Report at index {idx} is missing 'model' field")

        val_metrics = r.get("validation_metrics")
        if not isinstance(val_metrics, dict) or metric_key not in val_metrics:
            raise ModelComparisonError(
                f"Report for model '{model_name}' is missing validation metric '{metric_key}'"
            )

        try:
            metric_val = float(val_metrics[metric_key])
        except (ValueError, TypeError) as err:
            raise ModelComparisonError(
                f"Report for model '{model_name}' has non-numeric validation metric '{metric_key}': {val_metrics[metric_key]}"
            ) from err

        validated_candidates.append({
            "original_index": idx,
            "model": model_name,
            "metric_value": metric_val,
            "report": r,
        })

    # Sort candidates strictly by validation metric
    # Higher is better for R2; lower is better for RMSE, MAE, MAPE.
    reverse_sort = (metric_key == "r2")

    # Stable sort preserving input order on exact metric ties
    sorted_candidates = sorted(
        validated_candidates,
        key=lambda item: item["metric_value"],
        reverse=reverse_sort,
    )

    ranking = []
    for rank_idx, item in enumerate(sorted_candidates, start=1):
        rep = item["report"]
        rank_entry = {
            "rank": rank_idx,
            "model": item["model"],
            "validation_metrics": rep.get("validation_metrics", {}),
            "test_metrics": rep.get("test_metrics", {}),
        }
        # Copy optional model metadata if present
        for key in ("configuration", "weights", "component_validation_metrics", "component_test_metrics"):
            if key in rep:
                rank_entry[key] = rep[key]

        ranking.append(rank_entry)

    recommended_model = ranking[0]["model"]

    return {
        "selection_metric": metric_key,
        "selection_split": "validation",
        "ranking": ranking,
        "recommended_model": recommended_model,
    }
