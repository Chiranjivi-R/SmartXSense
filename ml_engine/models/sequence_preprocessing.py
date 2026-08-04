"""Sequence preprocessing for LSTM time-series forecasting.

Wraps the existing Phase 5E TabularFeatureTransformer and adds:
- Train-fitted StandardScaler for features
- Train-fitted StandardScaler for target (optional)
- Chronological sequence construction with configurable lookback
- Split-boundary history: validation/test sequences may use trailing
  context from the preceding partition (never future rows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .tabular_preprocessing import TabularFeatureTransformer


class SequencePreprocessingError(ValueError):
    """Raised when sequence construction or preprocessing fails."""


class SequencePreprocessor:
    """Train-only fitted preprocessor that produces (X_seq, y) pairs for LSTM.

    Leakage & Pipeline Rules:
    1. Reuses TabularFeatureTransformer for categorical encoding, imputation, and
       all-missing feature exclusion — all fitted on TRAIN only.
    2. Feature scaler (StandardScaler) fitted on TRAIN only.
    3. Target scaler (StandardScaler) fitted on TRAIN only.
    4. build_sequences() is chronological; never shuffles.
    5. current target y(t) is NEVER included in its own input sequence X(t).
    6. Validation/test sequence construction may prepend `history_df` rows as
       context — but only rows that chronologically precede the split boundary.
    """

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self._tabular_transformer: TabularFeatureTransformer | None = None
        self._feature_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._is_fitted = False
        self.feature_names: list[str] = []
        self.n_features: int = 0

    def fit(self, train_df: pd.DataFrame) -> SequencePreprocessor:
        """Fit all preprocessing components exclusively on training data."""
        if train_df is None or train_df.empty:
            raise SequencePreprocessingError("Training DataFrame cannot be empty")

        # Step 1: Fit tabular transformer (handles encoding, imputation, all-missing exclusion)
        self._tabular_transformer = TabularFeatureTransformer(target_column=self.target_column)
        self._tabular_transformer.fit(train_df)
        self.feature_names = self._tabular_transformer.feature_names
        self.n_features = len(self.feature_names)

        # Step 2: Fit feature scaler on train feature matrix
        X_train_raw = self._tabular_transformer.transform(train_df)
        self._feature_scaler = StandardScaler()
        self._feature_scaler.fit(X_train_raw)

        # Step 3: Fit target scaler on train target values
        y_train = train_df[self.target_column].astype(float).to_numpy().reshape(-1, 1)
        self._target_scaler = StandardScaler()
        self._target_scaler.fit(y_train)

        self._is_fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into a scaled 2D feature matrix.

        Uses train-fitted tabular transformer + feature scaler.
        """
        if not self._is_fitted or self._tabular_transformer is None or self._feature_scaler is None:
            raise SequencePreprocessingError("SequencePreprocessor must be fitted before calling transform_features")
        X_raw = self._tabular_transformer.transform(df)
        return self._feature_scaler.transform(X_raw)

    def scale_target(self, y: np.ndarray) -> np.ndarray:
        """Scale target values using train-fitted target scaler."""
        if self._target_scaler is None:
            raise SequencePreprocessingError("SequencePreprocessor must be fitted before scaling targets")
        return self._target_scaler.transform(y.reshape(-1, 1)).ravel()

    def inverse_scale_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled predictions back to original target units."""
        if self._target_scaler is None:
            raise SequencePreprocessingError("SequencePreprocessor must be fitted before inverse scaling")
        return self._target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def build_sequences(
        self,
        df: pd.DataFrame,
        lookback: int,
        history_df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build chronological (X_seq, y) pairs for LSTM training/evaluation.

        Args:
            df: DataFrame for which to produce sequences (train, val, or test split).
            lookback: Number of past time steps W used as input to each prediction.
            history_df: Optional preceding DataFrame (e.g., TRAIN when building
                        validation sequences) providing lookback context for the
                        first `lookback` rows. Must be chronologically before `df`.
                        No rows from `df` are used as future context for `df` rows.

        Returns:
            X_seq: shape (N, lookback, n_features) — scaled feature sequences
            y_seq: shape (N,) — scaled target values

        Sequence construction policy:
            X(t) = scaled features from rows (t-lookback) through (t-1)
            y(t) = scaled target at row t
            y(t) is NEVER in X(t).
        """
        if not self._is_fitted:
            raise SequencePreprocessingError("SequencePreprocessor must be fitted before building sequences")
        if lookback < 1:
            raise SequencePreprocessingError("lookback must be >= 1")

        # Transform the eval df and optionally prepend history
        X_eval = self.transform_features(df)
        y_eval = self.scale_target(df[self.target_column].astype(float).to_numpy())

        if history_df is not None and not history_df.empty:
            X_hist = self.transform_features(history_df)
            y_hist = self.scale_target(history_df[self.target_column].astype(float).to_numpy())
            X_combined = np.concatenate([X_hist, X_eval], axis=0)
            y_combined = np.concatenate([y_hist, y_eval], axis=0)
            offset = len(y_hist)  # prediction targets start at this index
        else:
            X_combined = X_eval
            y_combined = y_eval
            offset = 0

        n_total = len(y_combined)

        # Build sequences: for each target at index i (>= offset+lookback-1 if no history
        # or >= lookback if with history), X uses indices i-lookback..i-1
        sequences_X: list[np.ndarray] = []
        sequences_y: list[float] = []

        start = max(offset, lookback)
        for i in range(start, n_total):
            seq_x = X_combined[i - lookback : i]   # past W steps (NOT including i)
            seq_y = y_combined[i]                   # target at i
            if seq_x.shape[0] == lookback:
                sequences_X.append(seq_x)
                sequences_y.append(seq_y)

        if not sequences_X:
            raise SequencePreprocessingError(
                f"No sequences could be built with lookback={lookback} and "
                f"{len(df)} eval rows and {len(history_df) if history_df is not None else 0} history rows"
            )

        return np.stack(sequences_X, axis=0), np.array(sequences_y, dtype=np.float32)
