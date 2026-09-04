"""XGBoost Regressor supervised forecasting model implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .tabular_preprocessing import TabularFeatureTransformer


class XGBoostModelError(ValueError):
    """Raised when XGBoost model fitting or prediction fails."""


class XGBoostForecaster:
    """Supervised XGBoost Regressor forecaster for tabular time-series features.

    Strict Leakage & Pipeline Rules:
    1. Feature matrix X excludes target column, timestamp, and revenue leakage fields.
    2. Categorical features are encoded and missing values are imputed strictly using
       training-split statistics.
    3. Exposes model-native feature importances.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.04,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
        n_jobs: int = -1,
        target_transform: str | None = None,
        sample_weight_strategy: str | None = None,
    ) -> None:
        if target_transform not in (None, "log1p"):
            raise XGBoostModelError(f"Unsupported target_transform '{target_transform}'. Supported: None, 'log1p'")
        if sample_weight_strategy not in (None, "spike_sensitive"):
            raise XGBoostModelError(
                f"Unsupported sample_weight_strategy '{sample_weight_strategy}'. Supported: None, 'spike_sensitive'"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.target_transform = target_transform
        self.sample_weight_strategy = sample_weight_strategy

        self._transformer: TabularFeatureTransformer | None = None
        self._model: XGBRegressor | None = None
        self.target_column: str = ""

    def fit(self, train_df: pd.DataFrame, target_column: str) -> XGBoostForecaster:
        """Fit preprocessing transformer and XGBoost regressor on training dataset."""
        if train_df is None or train_df.empty:
            raise XGBoostModelError("Training dataset cannot be empty")
        if target_column not in train_df.columns:
            raise XGBoostModelError(f"Target column '{target_column}' missing from training data")

        self.target_column = target_column
        self._transformer = TabularFeatureTransformer(target_column=target_column)
        self._transformer.fit(train_df)

        X_train = self._transformer.transform(train_df)
        raw_y = train_df[target_column].astype(float).to_numpy()

        sample_weights = None
        if self.sample_weight_strategy == "spike_sensitive":
            if len(raw_y) >= 10:
                q85 = float(np.percentile(raw_y, 85))
                q95 = float(np.percentile(raw_y, 95))
                if q95 > q85:
                    sample_weights = np.ones(len(raw_y), dtype=float)
                    sample_weights[raw_y > q85] = 1.10
                    sample_weights[raw_y > q95] = 1.30

        y_train = raw_y.copy()
        if self.target_transform == "log1p":
            if np.any(y_train < -1.0):
                raise XGBoostModelError("Cannot apply log1p transform to target values < -1.0")
            y_train = np.log1p(np.maximum(y_train, 0.0))

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        try:
            self._model.fit(X_train, y_train, sample_weight=sample_weights)
        except Exception as err:
            raise XGBoostModelError(f"Failed to fit XGBRegressor: {err}") from err

        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        """Generate point forecasts for evaluation dataset matching index and length."""
        if self._model is None or self._transformer is None:
            raise XGBoostModelError("XGBoostForecaster must be fitted before calling predict")
        if eval_df is None or eval_df.empty:
            raise XGBoostModelError("Evaluation dataset cannot be empty")

        X_eval = self._transformer.transform(eval_df)
        try:
            preds = self._model.predict(X_eval)
            if self.target_transform == "log1p":
                preds = np.expm1(preds)
                preds = np.clip(preds, a_min=0.0, a_max=None)
        except Exception as err:
            raise XGBoostModelError(f"Failed to predict using XGBRegressor: {err}") from err

        return pd.Series(preds, index=eval_df.index, dtype=float)

    def get_feature_importances(self) -> dict[str, float]:
        """Return model-native feature importances mapping feature name to importance score."""
        if self._model is None or self._transformer is None:
            raise XGBoostModelError("XGBoostForecaster must be fitted to get feature importances")

        names = self._transformer.feature_names
        importances = self._model.feature_importances_

        importance_dict = {name: float(imp) for name, imp in zip(names, importances)}
        # Sort descending by importance
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    @property
    def feature_names(self) -> list[str]:
        if self._transformer is None:
            return []
        return self._transformer.feature_names
