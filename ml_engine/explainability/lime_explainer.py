"""LIME tabular explainer for tree-based forecasting models.

Strict Pipeline & Consistency Rules:
1. Uses TRAIN-fitted TabularFeatureTransformer to format feature inputs.
2. Generates local linear explanations without mutating underlying model weights.
3. Returns serializable standard Python structures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular


class LimeExplainerError(ValueError):
    """Raised when LIME explanation fails or input data is invalid."""


class LimeTabularExplainer:
    """LIME explainer for tree-based forecasters (RandomForestForecaster, XGBoostForecaster)."""

    def __init__(self, forecaster: Any, train_df: pd.DataFrame) -> None:
        if not hasattr(forecaster, "_model") or not hasattr(forecaster, "_transformer"):
            raise LimeExplainerError(
                "LimeTabularExplainer requires a fitted forecaster containing '_model' and '_transformer'."
            )
        if forecaster._model is None or forecaster._transformer is None:
            raise LimeExplainerError("Forecaster must be fitted before generating LIME explanations.")
        if train_df is None or train_df.empty:
            raise LimeExplainerError("Training DataFrame train_df is required for LIME background statistics.")

        self.forecaster = forecaster
        self.model = forecaster._model
        self.transformer = forecaster._transformer
        self.feature_names = list(self.transformer.feature_names)

        X_train = self.transformer.transform(train_df)

        try:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                mode="regression",
                random_state=42,
            )
        except Exception as err:
            raise LimeExplainerError(f"Failed to initialize LimeTabularExplainer: {err}") from err

    def explain_local(self, eval_df: pd.DataFrame, sample_index: int = 0) -> dict[str, Any]:
        """Compute local LIME feature attribution for a specific evaluation sample row."""
        if sample_index < 0 or sample_index >= len(eval_df):
            raise LimeExplainerError(
                f"Sample index {sample_index} out of range for evaluation data of size {len(eval_df)}."
            )

        X_trans = self.transformer.transform(eval_df)
        sample_row = X_trans[sample_index]

        def predict_fn(x: np.ndarray) -> np.ndarray:
            return self.model.predict(x)

        try:
            exp = self.explainer.explain_instance(
                data_row=sample_row,
                predict_fn=predict_fn,
                num_features=len(self.feature_names),
            )
        except Exception as err:
            raise LimeExplainerError(f"Failed to generate LIME explanation: {err}") from err

        exp_list = exp.as_list()
        intercept = float(exp.intercept[0]) if hasattr(exp, "intercept") and len(exp.intercept) > 0 else 0.0

        preds = self.forecaster.predict(eval_df.iloc[[sample_index]])
        prediction = float(preds.iloc[0])

        feature_attributions = [
            {
                "rule": condition,
                "weight": float(weight),
            }
            for condition, weight in exp_list
        ]

        return {
            "explanation_type": "local",
            "method": "lime",
            "sample_index": sample_index,
            "prediction": prediction,
            "intercept": intercept,
            "feature_attributions": feature_attributions,
        }
