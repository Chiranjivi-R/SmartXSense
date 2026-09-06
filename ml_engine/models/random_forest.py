"""Random Forest Regressor supervised forecasting model implementation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .tabular_preprocessing import TabularFeatureTransformer


RETAIL_MACRO_EXCLUSIONS: tuple[str, ...] = (
    "cpi_clothing",
    "cpi_combined",
    "cpi_food",
    "fuel_price_petrol",
    "gdp_growth_rate",
    "inflation_rate",
    "repo_rate",
    "unemployment_rate",
    "wpi_index",
)


class RandomForestModelError(ValueError):
    """Raised when Random Forest model fitting or prediction fails."""


class RandomForestForecaster:
    """Supervised Random Forest Regressor forecaster for tabular time-series features.

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
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        target_transform: str | None = None,
        feature_exclusions: Sequence[str] | None = None,
    ) -> None:
        if target_transform not in (None, "log1p"):
            raise RandomForestModelError(f"Unsupported target_transform '{target_transform}'. Supported: None, 'log1p'")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.target_transform = target_transform
        self.feature_exclusions = list(feature_exclusions) if feature_exclusions is not None else None

        self._transformer: TabularFeatureTransformer | None = None
        self._model: RandomForestRegressor | None = None
        self.target_column: str = ""

    def fit(self, train_df: pd.DataFrame, target_column: str) -> RandomForestForecaster:
        """Fit preprocessing transformer and Random Forest regressor on training dataset."""
        if train_df is None or train_df.empty:
            raise RandomForestModelError("Training dataset cannot be empty")
        if target_column not in train_df.columns:
            raise RandomForestModelError(f"Target column '{target_column}' missing from training data")

        self.target_column = target_column
        extra_exclude = self.feature_exclusions or ()
        self._transformer = TabularFeatureTransformer(target_column=target_column, extra_exclude=extra_exclude)
        self._transformer.fit(train_df)

        X_train = self._transformer.transform(train_df)
        y_train = train_df[target_column].astype(float).to_numpy()

        if self.target_transform == "log1p":
            if np.any(y_train < -1.0):
                raise RandomForestModelError("Cannot apply log1p transform to target values < -1.0")
            y_train = np.log1p(np.maximum(y_train, 0.0))

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        try:
            self._model.fit(X_train, y_train)
        except Exception as err:
            raise RandomForestModelError(f"Failed to fit RandomForestRegressor: {err}") from err

        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        """Generate point forecasts for evaluation dataset matching index and length."""
        if self._model is None or self._transformer is None:
            raise RandomForestModelError("RandomForestForecaster must be fitted before calling predict")
        if eval_df is None or eval_df.empty:
            raise RandomForestModelError("Evaluation dataset cannot be empty")

        X_eval = self._transformer.transform(eval_df)
        try:
            preds = self._model.predict(X_eval)
            if getattr(self, "target_transform", None) == "log1p":
                preds = np.expm1(preds)
                preds = np.clip(preds, a_min=0.0, a_max=None)
        except Exception as err:
            raise RandomForestModelError(f"Failed to predict using RandomForestRegressor: {err}") from err

        return pd.Series(preds, index=eval_df.index, dtype=float)

    def get_feature_importances(self) -> dict[str, float]:
        """Return model-native feature importances mapping feature name to importance score."""
        if self._model is None or self._transformer is None:
            raise RandomForestModelError("RandomForestForecaster must be fitted to get feature importances")

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
