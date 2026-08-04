"""XGBoost Regressor supervised forecasting model implementation."""

from __future__ import annotations

from typing import Any

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
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

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
        y_train = train_df[target_column].astype(float).to_numpy()

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
            self._model.fit(X_train, y_train)
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
