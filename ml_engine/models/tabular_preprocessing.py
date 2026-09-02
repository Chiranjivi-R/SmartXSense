"""Tabular feature selection, leakage-prevention, categorical encoding, and imputation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class PreprocessingError(ValueError):
    """Raised when tabular preprocessing fails."""


DEFAULT_EXCLUDED_COLUMNS = ("timestamp", "revenue")


class TabularFeatureTransformer:
    """Train-fitted transformer for time-series tabular ML models.

    Strict Leakage & Pipeline Rules:
    1. Target column, timestamp, and revenue leakage columns are excluded from feature matrix X.
    2. Categorical features are encoded using OrdinalEncoder fitted strictly on TRAIN data.
       Unseen categories in validation/test receive -1.
    3. Missing lag/rolling features are imputed using SimpleImputer(strategy="median")
       fitted strictly on TRAIN data.
    """

    def __init__(self, target_column: str, extra_exclude: Sequence[str] = ()) -> None:
        self.target_column = target_column
        self.excluded_columns = set(DEFAULT_EXCLUDED_COLUMNS) | {target_column} | set(extra_exclude)
        self.feature_names: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self._encoder: OrdinalEncoder | None = None
        self._imputer: SimpleImputer | None = None
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> TabularFeatureTransformer:
        """Fit categorical encoders and missing-value imputers on training data only."""
        if train_df is None or train_df.empty:
            raise PreprocessingError("Training DataFrame cannot be empty")

        candidates = [
            col
            for col in train_df.columns
            if col not in self.excluded_columns and not train_df[col].replace("", np.nan).dropna().empty
        ]


        if not candidates:
            self.feature_names = []
            self.categorical_columns = []
            self.numeric_columns = []
            self._imputer = None
            self._is_fitted = True
            return self

        self.feature_names = candidates
        self.categorical_columns = [
            col for col in candidates if train_df[col].dtype == "object" or train_df[col].dtype.name == "category"
        ]
        self.numeric_columns = [col for col in candidates if col not in self.categorical_columns]


        X_train = train_df[self.feature_names].copy()

        # Fit OrdinalEncoder for categorical columns
        if self.categorical_columns:
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            # Fill NaNs in text/cat columns with string placeholder before fitting encoder
            cat_data = X_train[self.categorical_columns].astype(str).fillna("__MISSING__")
            self._encoder.fit(cat_data)
            X_train[self.categorical_columns] = self._encoder.transform(cat_data)

        # Ensure numeric conversion and guarantee non-empty arrays for imputer
        for col in self.numeric_columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            if X_train[col].isna().all():
                X_train[col] = 0.0

        # Fit SimpleImputer on the full processed feature matrix
        self._imputer = SimpleImputer(strategy="median")
        self._imputer.fit(X_train.to_numpy(dtype=np.float64))

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform validation or test DataFrames using train-fitted parameters."""
        if not self._is_fitted:
            raise PreprocessingError("TabularFeatureTransformer must be fitted before calling transform")

        if df is None or df.empty:
            raise PreprocessingError("DataFrame to transform cannot be empty")

        if not self.feature_names:
            return np.zeros((len(df), 0), dtype=np.float64)


        missing_features = [col for col in self.feature_names if col not in df.columns]
        if missing_features:
            raise PreprocessingError(f"DataFrame is missing features learned during training: {missing_features}")

        X_eval = df[self.feature_names].copy()

        # Encode categorical columns using train-fitted encoder
        if self.categorical_columns and self._encoder is not None:
            cat_data = X_eval[self.categorical_columns].astype(str).fillna("__MISSING__")
            X_eval[self.categorical_columns] = self._encoder.transform(cat_data)

        for col in self.numeric_columns:
            X_eval[col] = pd.to_numeric(X_eval[col], errors="coerce")
            if X_eval[col].isna().all():
                X_eval[col] = 0.0

        # Impute missing values using train-fitted imputer
        matrix = X_eval.to_numpy(dtype=np.float64)
        return self._imputer.transform(matrix)
