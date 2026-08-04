"""ARIMA statistical forecasting model implementation."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


class ARIMAModelError(ValueError):
    """Raised when ARIMA model fitting or forecasting fails."""


class ARIMAForecaster:
    """Chronological ARIMA forecasting model using statsmodels.

    Strict Leakage & Evaluation Principles:
    1. Fits model parameters (p, d, q) ONLY on training target data.
    2. Validation and test target observations are never used for parameter estimation.
    3. Efficiently generates predictions for validation/test without per-step re-fitting.
    4. Configurable (p, d, q) order tuple defaulting to (1, 1, 1).
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)) -> None:
        self.order = self._validate_order(order)
        self._fitted_model: ARIMAResults | None = None
        self._last_train_target: pd.Series | None = None

    @staticmethod
    def _validate_order(order: Any) -> tuple[int, int, int]:
        if not isinstance(order, (tuple, list)) or len(order) != 3:
            raise ARIMAModelError("ARIMA order must be a 3-element tuple (p, d, q)")

        p, d, q = order
        for val, name in zip((p, d, q), ("p", "d", "q")):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ARIMAModelError(f"ARIMA parameter '{name}' must be a non-negative integer")

        return int(p), int(d), int(q)

    def fit(self, train_df: pd.DataFrame | pd.Series, target_column: str = "demand_mw") -> ARIMAForecaster:
        """Fit ARIMA model parameters strictly on training target values."""
        if train_df is None or len(train_df) == 0:
            raise ARIMAModelError("Training data is empty")

        if isinstance(train_df, pd.DataFrame):
            if target_column not in train_df.columns:
                raise ARIMAModelError(f"Target column '{target_column}' missing from training data")
            target_series = train_df[target_column]
        elif isinstance(train_df, pd.Series):
            target_series = train_df
        else:
            raise ARIMAModelError("train_df must be a pandas DataFrame or Series")

        if target_series.empty:
            raise ARIMAModelError("Training data is empty")

        clean_series = target_series.astype(float).reset_index(drop=True)
        if clean_series.isna().any():
            raise ARIMAModelError("Training target series contains NaN values")

        self._last_train_target = clean_series

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(clean_series, order=self.order)
                self._fitted_model = model.fit()
        except Exception as err:
            raise ARIMAModelError(f"Failed to fit ARIMA model with order {self.order}: {err}") from err

        return self

    def predict(
        self,
        eval_df: pd.DataFrame | pd.Series,
        target_column: str = "demand_mw",
        history: pd.Series | pd.DataFrame | None = None,
        dynamic: bool = False,
    ) -> pd.Series:
        """Generate forecasts for evaluation dataset matching timestamp alignment.

        If dynamic is False and eval target values are available, 1-step ahead
        state-space predictions are generated without re-estimating model parameters.
        If dynamic is True or eval target values are missing, out-of-sample recursive
        forecasts are produced.
        """
        if self._fitted_model is None:
            raise ARIMAModelError("ARIMA model must be fitted before calling predict")

        if isinstance(eval_df, pd.DataFrame):
            eval_series = eval_df[target_column] if target_column in eval_df.columns else None
            eval_index = eval_df.index
        elif isinstance(eval_df, pd.Series):
            eval_series = eval_df
            eval_index = eval_df.index
        else:
            raise ARIMAModelError("eval_df must be a pandas DataFrame or Series")

        n_eval = len(eval_df)
        if n_eval == 0:
            raise ARIMAModelError("Evaluation dataset is empty")

        # Determine reference history
        ref_history = self._last_train_target
        if history is not None:
            if isinstance(history, pd.DataFrame) and target_column in history.columns:
                ref_history = history[target_column].astype(float).reset_index(drop=True)
            elif isinstance(history, pd.Series):
                ref_history = history.astype(float).reset_index(drop=True)

        if not dynamic and eval_series is not None and ref_history is not None and not ref_history.empty:
            try:
                hist_clean = ref_history.astype(float).reset_index(drop=True)
                eval_clean = eval_series.astype(float).reset_index(drop=True)
                combined = pd.concat([hist_clean, eval_clean], ignore_index=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    applied = self._fitted_model.apply(combined, refit=False)
                    fitted = applied.fittedvalues
                pred_tail = fitted.iloc[len(hist_clean):].reset_index(drop=True)
                # Handle initial NaN if d > 0 in statsmodels fitted values
                if pred_tail.isna().any():
                    pred_tail = pred_tail.bfill().ffill()
                pred_tail.index = eval_index
                return pred_tail
            except Exception:
                # Fallback to out-of-sample dynamic forecast if apply fails
                pass

        # Dynamic multi-step out-of-sample forecast
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast_series = self._fitted_model.forecast(steps=n_eval)
            forecast_res = pd.Series(forecast_series.values, index=eval_index, dtype=float)
            if forecast_res.isna().any():
                forecast_res = forecast_res.bfill().ffill()
            return forecast_res
        except Exception as err:
            raise ARIMAModelError(f"Failed to generate ARIMA forecast: {err}") from err
