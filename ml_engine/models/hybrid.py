"""Validation-Driven Hybrid Forecasting Engine for SmartXSense.

Phase 5H implements a leakage-safe ensemble mechanism that combines predictions
from multiple base models using weights derived EXCLUSIVELY from validation performance.

Strict Leakage & Pipeline Rules:
1. Hybrid weights are computed ONLY from component VALIDATION metrics.
2. TEST set targets and metrics are NEVER used to calculate or optimize weights.
3. Component predictions must be aligned on the exact same timestamps/indices.
4. Frozen validation weights are applied to TEST predictions for final evaluation.
"""

from __future__ import annotations

from typing import Any
import itertools

import numpy as np
import pandas as pd

from ml_engine.evaluation.metrics import evaluate_forecast


class HybridModelError(ValueError):
    """Raised when hybrid forecasting initialization, weight calculation, or combination fails."""


SUPPORTED_COMPONENTS = (
    "naive",
    "seasonal_naive",
    "arima",
    "sarima",
    "random_forest",
    "xgboost",
    "trend_xgboost",
    "lstm",
    "cnn_lstm",
)

DEFAULT_HYBRID_COMPONENTS = {
    "electricity": ["arima", "random_forest", "xgboost"],
    "retail": ["random_forest", "xgboost"],
}


def generate_hybrid_candidates(
    domain: str,
    applicable_models: list[str] | None = None,
    future_forecast_only: bool = False,
    max_components: int = 3,
    horizon: int | None = None,
) -> list[list[str]]:
    """Generate a bounded, deterministic list of hybrid candidate compositions of size >= 2.

    Rules:
    - Bounded: size 2 up to min(max_components, len(pool)).
    - Domain-aware: rejects domain-incompatible models (e.g. ARIMA for retail).
    - Future-compatible filtering when future_forecast_only=True.
    - Preserves deterministic canonical sorting.
    """
    from .future_forecasting import is_future_forecast_supported

    domain_clean = domain.lower().strip()
    if applicable_models is None:
        raw_pool = DEFAULT_HYBRID_COMPONENTS.get(domain_clean, ["random_forest", "xgboost"])
    else:
        raw_pool = applicable_models

    pool: list[str] = []
    for m in raw_pool:
        m_clean = m.lower().strip()
        if m_clean == "hybrid":
            continue
        if m_clean not in SUPPORTED_COMPONENTS:
            continue
        if m_clean in ("arima", "sarima") and domain_clean == "retail":
            continue
        if future_forecast_only and not is_future_forecast_supported(domain_clean, m_clean, horizon=horizon):
            continue
        if m_clean not in pool:
            pool.append(m_clean)

    pool = sorted(pool)
    candidates: list[list[str]] = []
    max_k = min(max_components, len(pool))
    for k in range(2, max_k + 1):
        for comb in itertools.combinations(pool, k):
            candidates.append(list(comb))

    return candidates


class HybridForecaster:
    """Validation-Driven Hybrid Forecaster.

    Combines predictions from multiple component models using validation performance weights.
    """

    def __init__(
        self,
        components: list[str] | None = None,
        domain: str = "electricity",
        weighting_strategy: str = "inverse_rmse",
        epsilon: float = 1e-8,
    ) -> None:
        self.domain = domain.lower().strip()
        self.weighting_strategy = weighting_strategy.lower().strip()
        self.epsilon = epsilon

        if self.weighting_strategy not in ("inverse_rmse", "inverse_mae", "equal"):
            raise HybridModelError(
                f"Unsupported weighting strategy '{weighting_strategy}'. "
                "Supported strategies: 'inverse_rmse', 'inverse_mae', 'equal'."
            )

        raw_components = (
            components
            if components is not None
            else DEFAULT_HYBRID_COMPONENTS.get(self.domain, ["random_forest", "xgboost"])
        )
        self.components = self.validate_components(raw_components, self.domain)
        self.weights: dict[str, float] | None = None
        self.candidate_evaluations: list[dict[str, Any]] | None = None
        self.best_hybrid: dict[str, Any] | None = None
        self.best_standalone: dict[str, Any] | None = None
        self.selection_decision: dict[str, Any] | None = None

    @staticmethod
    def validate_components(components: list[str], domain: str) -> list[str]:
        """Validate and normalize the list of component model names.

        Rules:
        - Rejects fewer than 2 components.
        - Rejects 'hybrid' as a component.
        - Rejects unknown model names.
        - Rejects domain-incompatible models (e.g., ARIMA for retail).
        - Deduplicates components while preserving order.
        """
        domain_clean = domain.lower().strip()
        normalized: list[str] = []

        for comp in components:
            comp_clean = comp.lower().strip()
            if comp_clean == "hybrid":
                raise HybridModelError("Cannot use 'hybrid' as a component model of a hybrid ensemble.")
            if comp_clean not in SUPPORTED_COMPONENTS:
                raise HybridModelError(
                    f"Unsupported component model '{comp}'. "
                    f"Supported components: {list(SUPPORTED_COMPONENTS)}"
                )
            if comp_clean in ("arima", "sarima") and domain_clean == "retail":
                raise HybridModelError(f"{comp_clean.upper()} model is not supported for the 'retail' domain.")

            if comp_clean not in normalized:
                normalized.append(comp_clean)

        if len(normalized) < 2:
            raise HybridModelError(
                f"Hybrid ensemble requires at least 2 distinct component models, got {len(normalized)}."
            )

        return normalized

    @classmethod
    def calculate_weights(
        cls,
        validation_metrics_map: dict[str, dict[str, float]],
        weighting_strategy: str = "inverse_rmse",
        epsilon: float = 1e-8,
    ) -> dict[str, float]:
        """Derive normalized model weights from component VALIDATION metrics.

        Strategies:
        - 'inverse_rmse': weight_m = (1 / (RMSE_m + epsilon)) / sum_j (1 / (RMSE_j + epsilon))
        - 'inverse_mae': weight_m = (1 / (MAE_m + epsilon)) / sum_j (1 / (MAE_j + epsilon))
        - 'equal': weight_m = 1 / M

        Raises:
            HybridModelError if validation metrics are missing or strategy is invalid.
        """
        strategy = weighting_strategy.lower().strip()
        if strategy not in ("inverse_rmse", "inverse_mae", "equal"):
            raise HybridModelError(f"Unsupported weighting strategy '{weighting_strategy}'.")

        if not validation_metrics_map or len(validation_metrics_map) < 2:
            raise HybridModelError("At least 2 component validation metrics are required to calculate weights.")

        models = list(validation_metrics_map.keys())
        M = len(models)

        if strategy == "equal":
            w_val = 1.0 / M
            return {m: w_val for m in models}

        metric_key = "mae" if strategy == "inverse_mae" else "rmse"
        raw_weights: dict[str, float] = {}
        for model_name, metrics in validation_metrics_map.items():
            val = metrics.get(metric_key, metrics.get("rmse", 1.0))
            val = max(0.0, float(val))
            raw_weights[model_name] = 1.0 / (val + epsilon)

        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            return {m: 1.0 / M for m in models}

        normalized_weights = {m: round(float(raw_weights[m] / total_raw), 4) for m in models}
        # Re-normalize slightly if rounding caused sum != 1.0
        w_sum = sum(normalized_weights.values())
        if w_sum > 0:
            normalized_weights = {m: round(float(v / w_sum), 4) for m, v in normalized_weights.items()}

        return normalized_weights

    @staticmethod
    def select_adaptive_winner(
        component_validation_metrics: dict[str, dict[str, float]],
        hybrid_validation_metrics: dict[str, float],
        selection_metric: str = "rmse",
    ) -> dict[str, Any]:
        """Select using validation metric only; test data is intentionally absent."""
        metric_key = selection_metric.lower().strip()
        if not component_validation_metrics or metric_key not in hybrid_validation_metrics:
            metric_key = "rmse"

        reverse = (metric_key == "r2")
        best_single, best_metrics = min(
            component_validation_metrics.items(),
            key=lambda item: -float(item[1].get(metric_key, 0.0)) if reverse else float(item[1].get(metric_key, float("inf"))),
        )
        best_score = float(best_metrics.get(metric_key, 0.0))
        hybrid_score = float(hybrid_validation_metrics.get(metric_key, 0.0))

        if reverse:
            hybrid_wins = hybrid_score > best_score
        else:
            hybrid_wins = hybrid_score < best_score

        selected = "hybrid" if hybrid_wins else best_single
        reason = (
            f"Hybrid validation {metric_key.upper()} ({hybrid_score:.4f}) improved on best single model "
            f"{best_single} ({best_score:.4f}); Hybrid was selected."
            if hybrid_wins else
            f"Hybrid validation {metric_key.upper()} ({hybrid_score:.4f}) did not improve on best single model "
            f"{best_single} ({best_score:.4f}); {best_single} was selected."
        )
        return {
            "selection_split": "validation",
            "selection_metric": metric_key,
            "best_single_model": best_single,
            "best_single_validation_score": best_score,
            "hybrid_validation_score": hybrid_score,
            "selected_model": selected,
            "selection_reason": reason,
        }

    @classmethod
    def evaluate_hybrid_candidates(
        cls,
        candidates: list[list[str]],
        component_val_predictions: dict[str, pd.Series],
        val_targets: pd.Series,
        component_test_predictions: dict[str, pd.Series] | None = None,
        test_targets: pd.Series | None = None,
        domain: str = "electricity",
        weighting_strategy: str = "inverse_rmse",
        selection_metric: str = "rmse",
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate a list of candidate hybrid combinations using cached predictions.

        Strict Leakage & Pipeline Rules:
        - NEVER refits or retrains base models.
        - Calculates candidate weights ONLY from component validation predictions.
        - TEST set targets and metrics are computed strictly for reporting.
        - Ranks evaluated candidates deterministically by validation selection metric.
        """
        from .future_forecasting import is_future_forecast_supported

        domain_clean = domain.lower().strip()
        metric_key = selection_metric.lower().strip()
        reverse = (metric_key == "r2")

        evaluated: list[dict[str, Any]] = []
        for cand in candidates:
            # Verify all components exist in component_val_predictions
            if not all(m in component_val_predictions for m in cand):
                continue

            cand_val_preds = {m: component_val_predictions[m] for m in cand}
            cand_val_metrics: dict[str, dict[str, float]] = {}
            for m, pred in cand_val_preds.items():
                cand_val_metrics[m] = evaluate_forecast(val_targets.loc[pred.index], pred)

            weights = cls.calculate_weights(
                cand_val_metrics,
                weighting_strategy=weighting_strategy,
            )

            val_hybrid_pred = cls.combine(cand_val_preds, weights)
            val_metrics = evaluate_forecast(val_targets.loc[val_hybrid_pred.index], val_hybrid_pred)

            test_metrics = None
            test_hybrid_pred = None
            if component_test_predictions is not None and test_targets is not None:
                if all(m in component_test_predictions for m in cand):
                    cand_test_preds = {m: component_test_predictions[m] for m in cand}
                    test_hybrid_pred = cls.combine(cand_test_preds, weights)
                    test_metrics = evaluate_forecast(test_targets.loc[test_hybrid_pred.index], test_hybrid_pred)

            future_capable = all(is_future_forecast_supported(domain_clean, m, horizon=horizon) for m in cand)

            evaluated.append({
                "components": cand,
                "weights": weights,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_pred": val_hybrid_pred,
                "test_pred": test_hybrid_pred,
                "future_forecast_supported": future_capable,
                "metric_value": float(val_metrics.get(metric_key, 0.0)),
            })

        # Rank candidates by validation metric
        evaluated.sort(key=lambda item: item["metric_value"], reverse=reverse)
        for rank_idx, item in enumerate(evaluated, start=1):
            item["rank"] = rank_idx

        return evaluated

    @classmethod
    def select_best_hybrid_and_standalone(
        cls,
        evaluated_hybrids: list[dict[str, Any]],
        component_val_metrics: dict[str, dict[str, float]],
        selection_metric: str = "rmse",
        require_future_forecast_support: bool = False,
    ) -> dict[str, Any]:
        """Select best hybrid, best standalone, and overall validation winner.

        Selection Rule:
        IF best_hybrid.validation_metric < best_standalone.validation_metric:
            selected_model = "hybrid" (with winning components)
        ELSE:
            selected_model = best_standalone (simpler model wins on tie)
        """
        metric_key = selection_metric.lower().strip()
        reverse = (metric_key == "r2")

        eligible_hybrids = evaluated_hybrids
        if require_future_forecast_support:
            eligible_hybrids = [h for h in evaluated_hybrids if h.get("future_forecast_supported", False)]

        if not eligible_hybrids:
            raise HybridModelError("No eligible hybrid candidates available for selection.")

        sorted_hybrids = sorted(
            eligible_hybrids,
            key=lambda h: -float(h.get("metric_value", h["validation_metrics"].get(metric_key, 0.0)))
            if reverse
            else float(h.get("metric_value", h["validation_metrics"].get(metric_key, float("inf")))),
        )
        best_hybrid = sorted_hybrids[0]
        hybrid_score = float(best_hybrid["validation_metrics"].get(metric_key, 0.0))

        # Select best standalone
        best_single, best_single_metrics = min(
            component_val_metrics.items(),
            key=lambda item: -float(item[1].get(metric_key, 0.0)) if reverse else float(item[1].get(metric_key, float("inf"))),
        )
        single_score = float(best_single_metrics.get(metric_key, 0.0))

        if reverse:
            hybrid_wins = hybrid_score > single_score
        else:
            hybrid_wins = hybrid_score < single_score

        selected = "hybrid" if hybrid_wins else best_single
        reason = (
            f"Best Hybrid {best_hybrid['components']} validation {metric_key.upper()} ({hybrid_score:.4f}) "
            f"improved on best single model '{best_single}' ({single_score:.4f}); Hybrid was selected."
            if hybrid_wins else
            f"Best Hybrid {best_hybrid['components']} validation {metric_key.upper()} ({hybrid_score:.4f}) "
            f"did not improve on best single model '{best_single}' ({single_score:.4f}); '{best_single}' was selected."
        )

        def _clean_cand(c: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in c.items() if k not in ("val_pred", "test_pred")}

        clean_best_hybrid = _clean_cand(best_hybrid)
        clean_evaluated = [_clean_cand(h) for h in sorted_hybrids]

        return {
            "selection_split": "validation",
            "selection_metric": metric_key,
            "selected_model": selected,
            "best_hybrid": clean_best_hybrid,
            "best_standalone": {
                "model": best_single,
                "validation_metrics": best_single_metrics,
                "metric_value": single_score,
            },
            "selection_reason": reason,
            "evaluated_candidates": clean_evaluated,
            "_best_hybrid_raw": best_hybrid,
        }

    @classmethod
    def validate_prediction_alignment(
        cls, predictions_map: dict[str, pd.Series]
    ) -> pd.Index:
        """Ensure all component prediction Series are valid, non-empty, and share the exact same index.

        Returns:
            The common pd.Index across all component predictions.
        """
        if not predictions_map or len(predictions_map) < 2:
            raise HybridModelError("Predictions map must contain at least 2 component predictions.")

        first_model = list(predictions_map.keys())[0]
        first_pred = predictions_map[first_model]

        if not isinstance(first_pred, pd.Series):
            raise HybridModelError(f"Prediction for '{first_model}' must be a pandas Series.")

        if first_pred.empty:
            raise HybridModelError(f"Prediction for '{first_model}' cannot be empty.")

        if first_pred.isna().any() or np.isinf(first_pred.to_numpy()).any():
            raise HybridModelError(f"Prediction for '{first_model}' contains NaN or infinite values.")

        common_index = first_pred.index

        for model_name, pred in predictions_map.items():
            if not isinstance(pred, pd.Series):
                raise HybridModelError(f"Prediction for '{model_name}' must be a pandas Series.")
            if pred.empty:
                raise HybridModelError(f"Prediction for '{model_name}' cannot be empty.")
            if pred.isna().any() or np.isinf(pred.to_numpy()).any():
                raise HybridModelError(f"Prediction for '{model_name}' contains NaN or infinite values.")

            if not pred.index.equals(common_index):
                raise HybridModelError(
                    f"Prediction index mismatch for '{model_name}'. All components must share the exact same index."
                )

        return common_index

    @classmethod
    def combine(
        cls,
        predictions_map: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> pd.Series:
        """Compute the weighted average prediction across component models.

        hybrid_pred(t) = sum_m (weight_m * pred_m(t))
        """
        common_index = cls.validate_prediction_alignment(predictions_map)

        if set(predictions_map.keys()) != set(weights.keys()):
            raise HybridModelError(
                f"Mismatch between prediction components ({list(predictions_map.keys())}) "
                f"and weight keys ({list(weights.keys())})."
            )

        combined_values = np.zeros(len(common_index), dtype=float)
        for model_name, pred in predictions_map.items():
            w = weights[model_name]
            combined_values += w * pred.to_numpy()

        return pd.Series(combined_values, index=common_index, dtype=float)

    def fit_and_combine(
        self,
        val_predictions: dict[str, pd.Series],
        val_targets: pd.Series,
        test_predictions: dict[str, pd.Series],
    ) -> tuple[dict[str, float], pd.Series, pd.Series, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Fit hybrid weights on VALIDATION metrics, combine val and test predictions.

        Returns:
            - weights
            - val_hybrid_pred
            - test_hybrid_pred
            - component_val_metrics
            - component_test_metrics
        """
        # Validate alignment
        cls = self.__class__
        val_index = cls.validate_prediction_alignment(val_predictions)
        test_index = cls.validate_prediction_alignment(test_predictions)

        # Evaluate component validation performance
        component_val_metrics: dict[str, dict[str, float]] = {}
        for m, pred in val_predictions.items():
            component_val_metrics[m] = evaluate_forecast(val_targets.loc[pred.index], pred)

        # Calculate and freeze weights using ONLY validation metrics
        self.weights = cls.calculate_weights(
            component_val_metrics,
            weighting_strategy=self.weighting_strategy,
            epsilon=self.epsilon,
        )

        # Combine predictions using frozen weights
        val_hybrid = cls.combine(val_predictions, self.weights)
        test_hybrid = cls.combine(test_predictions, self.weights)

        return self.weights, val_hybrid, test_hybrid, component_val_metrics
