"""ForecastExplainer facade for generating model explanations.

Isolates XAI logic from training and evaluation pipelines.
Supports SHAP (global & local) and LIME (local) for tree-based forecasters.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .lime_explainer import LimeExplainerError, LimeTabularExplainer
from .shap_explainer import ShapExplainerError, ShapTreeExplainer


class ExplainabilityError(ValueError):
    """Raised when explanation request is invalid or attempted on an unsupported model."""


SUPPORTED_EXPLAINABLE_MODELS = ("random_forest", "xgboost")


class ForecastExplainer:
    """Facade for generating model-level and feature-level explanations."""

    def __init__(self, forecaster: Any, model_name: str) -> None:
        self.model_name = model_name.lower().strip()
        if self.model_name not in SUPPORTED_EXPLAINABLE_MODELS:
            raise ExplainabilityError(
                f"Explainability is currently supported for Random Forest and XGBoost. "
                f"Received unsupported model '{model_name}'."
            )
        if forecaster is None or not hasattr(forecaster, "_model") or forecaster._model is None:
            raise ExplainabilityError(f"Fitted forecaster is required for model '{model_name}'.")

        self.forecaster = forecaster
        self.shap_explainer = ShapTreeExplainer(forecaster)

    def explain_global(self, eval_df: pd.DataFrame, method: str = "shap") -> dict[str, Any]:
        """Generate global feature importance explanation."""
        method_clean = method.lower().strip()
        if method_clean != "shap":
            raise ExplainabilityError(f"Global explanation method '{method}' is not supported. Use 'shap'.")

        try:
            res = self.shap_explainer.explain_global(eval_df)
            res["model"] = self.model_name
            return res
        except ShapExplainerError as err:
            raise ExplainabilityError(f"Global SHAP explanation failed: {err}") from err

    def explain_local(
        self,
        eval_df: pd.DataFrame,
        sample_index: int = 0,
        method: str = "shap",
        train_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate local sample attribution explanation."""
        method_clean = method.lower().strip()
        if method_clean == "shap":
            try:
                res = self.shap_explainer.explain_local(eval_df, sample_index=sample_index)
                res["model"] = self.model_name
                return res
            except ShapExplainerError as err:
                raise ExplainabilityError(f"Local SHAP explanation failed: {err}") from err
        elif method_clean == "lime":
            if train_df is None or train_df.empty:
                raise ExplainabilityError("LIME local explanation requires training dataset 'train_df'.")
            try:
                lime_exp = LimeTabularExplainer(self.forecaster, train_df=train_df)
                res = lime_exp.explain_local(eval_df, sample_index=sample_index)
                res["model"] = self.model_name
                return res
            except LimeExplainerError as err:
                raise ExplainabilityError(f"Local LIME explanation failed: {err}") from err
        else:
            raise ExplainabilityError(f"Unsupported explanation method '{method}'. Supported methods: 'shap', 'lime'.")
