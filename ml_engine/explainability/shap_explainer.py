"""SHAP tree explainer for tree-based forecasting models (Random Forest and XGBoost).

Strict Pipeline & Consistency Rules:
1. Feature matrix X is generated using the model's TRAIN-fitted TabularFeatureTransformer.
2. Preprocessing parameters (encoders, imputers) are NOT refitted on evaluation data.
3. Feature names align 1-to-1 with model feature matrix dimensions.
4. Outputs are fully JSON-serializable standard Python types.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


class ShapExplainerError(ValueError):
    """Raised when SHAP explanation fails or is attempted on unsupported models."""


class ShapTreeExplainer:
    """SHAP explainer for tree-based forecasters (RandomForestForecaster, XGBoostForecaster)."""

    def __init__(self, forecaster: Any) -> None:
        if not hasattr(forecaster, "_model") or not hasattr(forecaster, "_transformer"):
            raise ShapExplainerError(
                "ShapTreeExplainer requires a fitted forecaster containing '_model' and '_transformer'."
            )
        if forecaster._model is None or forecaster._transformer is None:
            raise ShapExplainerError("Forecaster must be fitted before generating explanations.")

        self.forecaster = forecaster
        self.model = forecaster._model
        self.transformer = forecaster._transformer
        self.feature_names = list(self.transformer.feature_names)

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as err:
            raise ShapExplainerError(f"Failed to initialize shap.TreeExplainer: {err}") from err

    def _get_transformed_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame using the forecaster's TRAIN-fitted transformer."""
        if df is None or df.empty:
            raise ShapExplainerError("Input DataFrame for explanation cannot be empty.")

        X_trans = self.transformer.transform(df)
        if X_trans.shape[1] != len(self.feature_names):
            raise ShapExplainerError(
                f"Feature dimension mismatch: transformer produced {X_trans.shape[1]} features, "
                f"expected {len(self.feature_names)}."
            )
        return X_trans

    def explain_global(self, eval_df: pd.DataFrame, max_samples: int = 100) -> dict[str, Any]:
        """Compute global feature importances based on mean absolute SHAP values."""
        sample_df = eval_df
        if len(eval_df) > max_samples:
            sample_df = eval_df.iloc[-max_samples:]

        X_trans = self._get_transformed_features(sample_df)
        try:
            shap_values = self.explainer.shap_values(X_trans)
        except Exception as err:
            raise ShapExplainerError(f"Failed to calculate SHAP values: {err}") from err

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        importance_dict = {
            name: float(val)
            for name, val in zip(self.feature_names, mean_abs)
        }
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        features_list = [
            {"feature": name, "importance": float(val)}
            for name, val in sorted_importance.items()
        ]

        return {
            "status": "success",
            "explanation_type": "global",
            "method": "shap",
            "sample_count": len(eval_df),
            "feature_count": len(self.feature_names),
            "feature_importance": sorted_importance,
            "features": features_list,
        }

    def explain_local(self, eval_df: pd.DataFrame, sample_index: int = 0) -> dict[str, Any]:
        """Compute local feature attribution for a specific sample row."""
        if sample_index < 0 or sample_index >= len(eval_df):
            raise ShapExplainerError(
                f"Sample index {sample_index} out of range for evaluation data of size {len(eval_df)}."
            )

        X_trans = self._get_transformed_features(eval_df)
        row_trans = X_trans[sample_index : sample_index + 1]

        try:
            shap_values = self.explainer.shap_values(row_trans)
        except Exception as err:
            raise ShapExplainerError(f"Failed to calculate local SHAP values: {err}") from err

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        sample_shap = shap_values[0]

        # Base value extraction
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[0])
        else:
            base_value = float(expected_val)

        # Get point prediction
        preds = self.forecaster.predict(eval_df.iloc[[sample_index]])
        prediction = float(preds.iloc[0])

        feature_values = {}
        contributions = {}
        items = []

        row_df = eval_df.iloc[sample_index]
        for f_idx, name in enumerate(self.feature_names):
            val = row_trans[0, f_idx]
            contrib = float(sample_shap[f_idx])
            raw_val = row_df[name] if name in row_df else val
            
            # Format feature value for JSON
            if pd.isna(raw_val):
                formatted_val = None
            elif isinstance(raw_val, (int, float, np.integer, np.floating)):
                formatted_val = float(raw_val)
            else:
                formatted_val = str(raw_val)

            feature_values[name] = formatted_val
            contributions[name] = contrib
            items.append({
                "feature": name,
                "contribution": contrib,
                "value": formatted_val,
            })

        top_positive = sorted(
            [item for item in items if item["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True,
        )
        top_negative = sorted(
            [item for item in items if item["contribution"] < 0],
            key=lambda x: x["contribution"],
        )

        return {
            "explanation_type": "local",
            "method": "shap",
            "sample_index": sample_index,
            "prediction": prediction,
            "base_value": base_value,
            "feature_values": feature_values,
            "shap_contributions": contributions,
            "top_positive_contributors": top_positive,
            "top_negative_contributors": top_negative,
        }
