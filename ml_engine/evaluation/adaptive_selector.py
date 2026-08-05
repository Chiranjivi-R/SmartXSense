"""Adaptive model selection layer based EXCLUSIVELY on validation metrics.

Strict Leakage & Pipeline Rules:
1. The adaptive selector selects the candidate model with the best VALIDATION metric.
2. TEST metrics are strictly IGNORED during model selection.
3. Deterministic tie-breaking behavior (preserves input report order on exact metric ties).
"""

from __future__ import annotations

from typing import Any

from .model_comparison import ModelComparisonError, compare_models


class AdaptiveSelectorError(ValueError):
    """Raised when adaptive model selection fails."""


class AdaptiveModelSelector:
    """Adaptive model selector for choosing optimal forecasting models based on validation performance."""

    def __init__(self, selection_metric: str = "rmse") -> None:
        self.selection_metric = selection_metric.lower().strip()

    def select_best_model(
        self,
        reports: list[dict[str, Any]],
        selection_metric: str | None = None,
    ) -> dict[str, Any]:
        """Select the best candidate model based strictly on validation performance metrics.

        Parameters
        ----------
        reports : list[dict[str, Any]]
            List of candidate model report dictionaries.
        selection_metric : str | None
            Override default selection metric ('rmse', 'mae', 'mape', 'r2').

        Returns
        -------
        dict[str, Any]
            Structured JSON dictionary containing selected_model, selection_metric,
            selection_split="validation", reason, and validation_metrics.
        """
        metric = (selection_metric if selection_metric is not None else self.selection_metric).lower().strip()
        try:
            comparison = compare_models(reports, selection_metric=metric)
        except ModelComparisonError as err:
            raise AdaptiveSelectorError(f"Adaptive model selection failed: {err}") from err

        top_rank = comparison["ranking"][0]
        selected_model = top_rank["model"]
        val_metrics = top_rank["validation_metrics"]
        metric_val = val_metrics.get(metric, 0.0)

        reason = (
            f"Selected model '{selected_model}' with optimal validation {metric.upper()} of {metric_val:.4f} "
            f"among {len(reports)} candidate models."
        )

        return {
            "selected_model": selected_model,
            "selection_metric": metric,
            "selection_split": "validation",
            "reason": reason,
            "validation_metrics": val_metrics,
        }
