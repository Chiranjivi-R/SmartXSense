"""Baseline forecasting models (Naive and Seasonal Naive)."""

from __future__ import annotations

import pandas as pd


class BaselineModelError(ValueError):
    """Raised when baseline forecasting cannot be executed."""


class NaiveForecaster:
    """Naive and Seasonal Naive persistent forecaster.

    - When seasonal_period = 1: Naive persistence forecast prediction(t) = target(t-1)
    - When seasonal_period = 24: Electricity seasonal persistence forecast prediction(t) = target(t-24)
    - When seasonal_period = 7: Retail seasonal persistence forecast prediction(t) = target(t-7)
    """

    def __init__(self, seasonal_period: int = 1) -> None:
        if not isinstance(seasonal_period, int) or isinstance(seasonal_period, bool) or seasonal_period < 1:
            raise BaselineModelError("seasonal_period must be a positive integer >= 1")
        self.seasonal_period = seasonal_period
        self._last_train_target: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame | pd.Series, target_column: str = "target") -> NaiveForecaster:
        """Fit model by storing the trailing training target history."""
        if train_df is None or len(train_df) == 0:
            raise BaselineModelError("Training data is empty")

        if isinstance(train_df, pd.DataFrame):
            if target_column not in train_df.columns:
                raise BaselineModelError(f"Target column '{target_column}' missing from training data")
            target_series = train_df[target_column]
        elif isinstance(train_df, pd.Series):
            target_series = train_df
        else:
            raise BaselineModelError("train_df must be a pandas DataFrame or Series")

        if target_series.empty:
            raise BaselineModelError("Training data is empty")

        self._last_train_target = target_series.astype(float).reset_index(drop=True)
        return self

    def predict(
        self,
        eval_df: pd.DataFrame | pd.Series,
        target_column: str = "target",
        history: pd.Series | pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate point forecasts for evaluation set.

        Predictions strictly rely on past target observations:
        prediction(t) = target(t - seasonal_period).
        """
        if isinstance(eval_df, pd.DataFrame):
            eval_series = eval_df[target_column] if target_column in eval_df.columns else None
            eval_index = eval_df.index
        elif isinstance(eval_df, pd.Series):
            eval_series = eval_df
            eval_index = eval_df.index
        else:
            raise BaselineModelError("eval_df must be a pandas DataFrame or Series")

        n_eval = len(eval_df)
        if n_eval == 0:
            raise BaselineModelError("Evaluation dataset is empty")

        # Determine reference history
        ref_history: pd.Series | None = None
        if history is not None:
            if isinstance(history, pd.DataFrame):
                ref_history = history[target_column] if target_column in history.columns else None
            elif isinstance(history, pd.Series):
                ref_history = history
        if ref_history is None:
            ref_history = self._last_train_target

        # Check if lag column is present in DataFrame (e.g. Phase 5C pre-computed features lag_1 or lag_24)
        lag_col_name = f"lag_{self.seasonal_period}"
        if isinstance(eval_df, pd.DataFrame) and lag_col_name in eval_df.columns and eval_df[lag_col_name].notna().all():
            predictions = eval_df[lag_col_name].astype(float).copy()
            predictions.index = eval_index
            return predictions

        # Otherwise construct sequence using history + eval targets
        if ref_history is not None and not ref_history.empty and eval_series is not None:
            hist_series = ref_history.astype(float).reset_index(drop=True)
            combined = pd.concat([hist_series, eval_series.astype(float).reset_index(drop=True)], ignore_index=True)
            # prediction at position i of combined is combined[i - seasonal_period]
            shifted = combined.shift(self.seasonal_period)
            # Backfill any initial NaNs using first available value
            shifted = shifted.bfill().ffill()
            pred_tail = shifted.iloc[len(hist_series):].reset_index(drop=True)
            pred_tail.index = eval_index
            return pred_tail

        if ref_history is not None and not ref_history.empty:
            # Out-of-sample static prediction using last seasonal_period values of history repeated
            hist_series = ref_history.astype(float).reset_index(drop=True)
            tail = hist_series.tail(self.seasonal_period).values
            repeated = [tail[i % self.seasonal_period] for i in range(n_eval)]
            return pd.Series(repeated, index=eval_index, dtype=float)

        raise BaselineModelError("No training history or target column available to generate baseline forecasts")
