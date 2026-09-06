"""XGBoost Regressor supervised forecasting model implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .tabular_preprocessing import TabularFeatureTransformer


class XGBoostModelError(ValueError):
    """Raised when XGBoost model fitting or prediction fails."""


CANONICAL_DETREND_LAGS = (
    ("lag_1", 1),
    ("lag_24", 24),
    ("lag_48", 48),
    ("lag_168", 168),
)


class XGBoostForecaster:
    """Supervised XGBoost Regressor forecaster for tabular time-series features.

    Strict Leakage & Pipeline Rules:
    1. Feature matrix X excludes target column, timestamp, and revenue leakage fields.
    2. Categorical features are encoded and missing values are imputed strictly using
       training-split statistics.
    3. Exposes model-native feature importances.
    4. Supports optional causal linear detrending (fitted strictly on training timestamps),
       detrending corresponding lag features and reconstructing predictions into original units.
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
        trend_mode: str | None = None,
        trend_knot_hour: float | None = None,
        residual_ar: bool = False,
        residual_ar_weights: dict[str, float] | None = None,
        residual_ar_lags: tuple[int, ...] = (1, 24, 168),
    ) -> None:
        if target_transform not in (None, "log1p"):
            raise XGBoostModelError(f"Unsupported target_transform '{target_transform}'. Supported: None, 'log1p'")
        if sample_weight_strategy not in (None, "spike_sensitive"):
            raise XGBoostModelError(
                f"Unsupported sample_weight_strategy '{sample_weight_strategy}'. Supported: None, 'spike_sensitive'"
            )
        if trend_mode in ("none", ""):
            trend_mode = None
        if trend_mode not in (None, "linear", "piecewise_linear"):
            raise XGBoostModelError(
                f"Unsupported trend_mode '{trend_mode}'. Supported: None, 'linear', 'piecewise_linear'"
            )
        if trend_knot_hour is not None:
            if not isinstance(trend_knot_hour, (int, float)) or np.isnan(trend_knot_hour) or trend_knot_hour <= 0:
                raise XGBoostModelError(f"trend_knot_hour must be a positive float, got '{trend_knot_hour}'")

        if residual_ar:
            if target_transform == "log1p":
                raise XGBoostModelError("Residual AR correction is only supported for linear scale (electricity)")
            if residual_ar_weights is not None:
                if not isinstance(residual_ar_weights, dict):
                    raise XGBoostModelError("residual_ar_weights must be a dict mapping lag names to float weights")
                sum_w = sum(abs(float(v)) for v in residual_ar_weights.values())
                if sum_w >= 1.0:
                    raise XGBoostModelError(
                        f"Residual AR weights sum to {sum_w:.4f} >= 1.0 (strictly non-stationary)"
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
        self.trend_mode = trend_mode
        self.trend_knot_hour = (
            float(trend_knot_hour)
            if trend_knot_hour is not None
            else (30000.0 if trend_mode == "piecewise_linear" else None)
        )
        self.residual_ar = bool(residual_ar)
        self.residual_ar_weights = dict(residual_ar_weights) if residual_ar_weights is not None else None
        self.residual_ar_lags = tuple(int(lag) for lag in residual_ar_lags)
        self.tail_residuals: np.ndarray | None = None

        self.trend_t0: pd.Timestamp | None = None
        self.trend_intercept: float = 0.0
        self.trend_slope: float = 0.0
        self.trend_slope2_delta: float = 0.0

        self._transformer: TabularFeatureTransformer | None = None
        self._model: XGBRegressor | None = None
        self.target_column: str = ""

    def compute_trend(self, t_hours: np.ndarray | float) -> np.ndarray | float:
        """Compute deterministic trend in MW for given hours elapsed from trend_t0."""
        if self.trend_mode == "linear":
            return self.trend_intercept + self.trend_slope * t_hours
        if self.trend_mode == "piecewise_linear":
            knot = self.trend_knot_hour if self.trend_knot_hour is not None else 30000.0
            return (
                self.trend_intercept
                + self.trend_slope * t_hours
                + self.trend_slope2_delta * np.maximum(0.0, t_hours - knot)
            )
        return 0.0

    def detrend_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detrend canonical lag columns against the historical trend offsets."""
        if self.trend_mode not in ("linear", "piecewise_linear") or self.trend_t0 is None:
            return df
        if "timestamp" not in df.columns:
            return df
        t_series = pd.to_datetime(df["timestamp"], utc=True)
        t_hours = (t_series - self.trend_t0).dt.total_seconds().to_numpy() / 3600.0

        df_proc = df.copy()
        for col, lag_h in CANONICAL_DETREND_LAGS:
            if col in df_proc.columns:
                t_lag = np.maximum(0.0, t_hours - lag_h)
                trend_lag = self.compute_trend(t_lag)
                df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce") - trend_lag
        return df_proc

    @property
    def trend_params(self) -> dict[str, Any] | None:
        if self.trend_mode not in ("linear", "piecewise_linear") or self.trend_t0 is None:
            return None
        params: dict[str, Any] = {
            "trend_mode": self.trend_mode,
            "slope": self.trend_slope,
            "intercept": self.trend_intercept,
            "t0": self.trend_t0,
        }
        if self.trend_mode == "piecewise_linear":
            params["trend_knot_hour"] = self.trend_knot_hour
            params["slope2_delta"] = self.trend_slope2_delta
            params["seg1_slope"] = self.trend_slope
            params["seg2_slope"] = self.trend_slope + self.trend_slope2_delta
        return params

    def fit(self, train_df: pd.DataFrame, target_column: str) -> XGBoostForecaster:
        """Fit preprocessing transformer and XGBoost regressor on training dataset."""
        if train_df is None or train_df.empty:
            raise XGBoostModelError("Training dataset cannot be empty")
        if target_column not in train_df.columns:
            raise XGBoostModelError(f"Target column '{target_column}' missing from training data")

        self.target_column = target_column
        raw_y = train_df[target_column].astype(float).to_numpy()

        df_train_proc = train_df
        y_train = raw_y.copy()

        if self.trend_mode in ("linear", "piecewise_linear"):
            if "timestamp" not in train_df.columns:
                raise XGBoostModelError(f"Timestamp column required for trend_mode='{self.trend_mode}'")
            t_series = pd.to_datetime(train_df["timestamp"], utc=True)
            self.trend_t0 = t_series.min()
            t_hours = (t_series - self.trend_t0).dt.total_seconds().to_numpy() / 3600.0

            if self.trend_mode == "linear":
                b, a = np.polyfit(t_hours, raw_y, 1)
                self.trend_intercept = float(a)
                self.trend_slope = float(b)
                self.trend_slope2_delta = 0.0
            elif self.trend_mode == "piecewise_linear":
                knot = self.trend_knot_hour if self.trend_knot_hour is not None else 30000.0
                X_trend = np.column_stack([np.ones_like(t_hours), t_hours, np.maximum(0.0, t_hours - knot)])
                coef, _, _, _ = np.linalg.lstsq(X_trend, raw_y, rcond=None)
                self.trend_intercept = float(coef[0])
                self.trend_slope = float(coef[1])
                self.trend_slope2_delta = float(coef[2])

            y_trend = self.compute_trend(t_hours)
            y_train = raw_y - y_trend

            df_train_proc = self.detrend_lag_features(train_df)
            df_train_proc[target_column] = y_train

        self._transformer = TabularFeatureTransformer(target_column=target_column)
        self._transformer.fit(df_train_proc)
        X_train = self._transformer.transform(df_train_proc)

        sample_weights = None
        if self.sample_weight_strategy == "spike_sensitive":
            target_for_weights = y_train
            if len(target_for_weights) >= 10:
                q85 = float(np.percentile(target_for_weights, 85))
                q95 = float(np.percentile(target_for_weights, 95))
                if q95 > q85:
                    sample_weights = np.ones(len(target_for_weights), dtype=float)
                    sample_weights[target_for_weights > q85] = 1.10
                    sample_weights[target_for_weights > q95] = 1.30

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

        if self.residual_ar:
            pred_train_detrended = self._model.predict(X_train)
            t_hours_tr = (
                (pd.to_datetime(train_df["timestamp"], utc=True) - self.trend_t0).dt.total_seconds().to_numpy() / 3600.0
                if self.trend_mode in ("linear", "piecewise_linear") and self.trend_t0 is not None
                else 0.0
            )
            trend_train = self.compute_trend(t_hours_tr) if self.trend_mode in ("linear", "piecewise_linear") else 0.0
            pred_total_tr = trend_train + pred_train_detrended
            res_tr = raw_y - pred_total_tr
            self.tail_residuals = res_tr[-168:].copy()

            if self.residual_ar_weights is None:
                N_tr = len(res_tr)
                if N_tr > 168:
                    X_ar = np.column_stack([
                        res_tr[168 - 1 : N_tr - 1],
                        res_tr[168 - 24 : N_tr - 24],
                        res_tr[168 - 168 : N_tr - 168],
                    ])
                    y_ar = res_tr[168:]
                    coef, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
                    sum_w = float(np.sum(np.abs(coef)))
                    if sum_w >= 1.0:
                        coef = coef * (0.95 / sum_w)
                    self.residual_ar_weights = {
                        "lag_1": float(coef[0]),
                        "lag_24": float(coef[1]),
                        "lag_168": float(coef[2]),
                    }
                else:
                    self.residual_ar_weights = {"lag_1": 0.0, "lag_24": 0.0, "lag_168": 0.0}

        return self

    def predict_components(
        self,
        eval_df: pd.DataFrame,
        *,
        with_residual_ar: bool = True,
        residual_history: Any = None,
    ) -> dict[str, np.ndarray]:
        """Generate breakdown of forecasts into trend, XGBoost residual, and AR error correction components."""
        if self._model is None or self._transformer is None:
            raise XGBoostModelError("XGBoostForecaster must be fitted before calling predict")
        if eval_df is None or eval_df.empty:
            raise XGBoostModelError("Evaluation dataset cannot be empty")

        df_eval_proc = eval_df
        eval_trend = np.zeros(len(eval_df), dtype=float)

        if self.trend_mode in ("linear", "piecewise_linear"):
            if "timestamp" not in eval_df.columns:
                raise XGBoostModelError(f"Timestamp column required for trend_mode='{self.trend_mode}'")
            if self.trend_t0 is None:
                raise XGBoostModelError("Model trend baseline timestamp is missing")

            t_eval = pd.to_datetime(eval_df["timestamp"], utc=True)
            t_hours = (t_eval - self.trend_t0).dt.total_seconds().to_numpy() / 3600.0

            df_eval_proc = self.detrend_lag_features(eval_df)
            eval_trend = np.asarray(self.compute_trend(t_hours), dtype=float)

        X_eval = self._transformer.transform(df_eval_proc)
        try:
            xgb_residual = np.asarray(self._model.predict(X_eval), dtype=float)
            if self.target_transform == "log1p":
                xgb_residual = np.expm1(xgb_residual)
                xgb_residual = np.clip(xgb_residual, a_min=0.0, a_max=None)
        except Exception as err:
            raise XGBoostModelError(f"Failed to predict using XGBRegressor: {err}") from err

        base_preds = eval_trend + xgb_residual
        ar_correction = np.zeros(len(eval_df), dtype=float)

        if self.residual_ar and with_residual_ar and self.residual_ar_weights is not None:
            w1 = float(self.residual_ar_weights.get("lag_1", 0.0))
            w24 = float(self.residual_ar_weights.get("lag_24", 0.0))
            w168 = float(self.residual_ar_weights.get("lag_168", 0.0))

            # Determine initial 168-lag residual buffer
            if residual_history is not None:
                init_tail = np.asarray(residual_history, dtype=float)[-168:]
            elif self.tail_residuals is not None and len(self.tail_residuals) > 0:
                init_tail = np.asarray(self.tail_residuals, dtype=float)[-168:]
            else:
                init_tail = np.zeros(168, dtype=float)

            if len(init_tail) < 168:
                init_tail = np.pad(init_tail, (168 - len(init_tail), 0), mode="constant")

            # Causal 1-step evaluation when actual targets are present
            if (
                self.target_column in eval_df.columns
                and not eval_df[self.target_column].isna().all()
                and len(eval_df) > 1
            ):
                actual_y = eval_df[self.target_column].to_numpy(dtype=float)
                eval_residuals = actual_y - base_preds
                full_res = np.concatenate([init_tail, eval_residuals])
                n_full = len(full_res)
                ar_correction = (
                    w1 * full_res[168 - 1 : n_full - 1]
                    + w24 * full_res[168 - 24 : n_full - 24]
                    + w168 * full_res[168 - 168 : n_full - 168]
                )
            else:
                # Recursive forward simulation when targets are absent (future forecast or single step)
                res_buf = list(init_tail[-168:])
                ar_corr_list = []
                for _ in range(len(eval_df)):
                    e_hat = float(w1 * res_buf[-1] + w24 * res_buf[-24] + w168 * res_buf[-168])
                    ar_corr_list.append(e_hat)
                    res_buf.append(e_hat)
                ar_correction = np.array(ar_corr_list, dtype=float)

        total_preds = base_preds + ar_correction
        return {
            "trend": eval_trend,
            "xgb_residual": xgb_residual,
            "ar_correction": ar_correction,
            "total": total_preds,
        }

    def predict(
        self,
        eval_df: pd.DataFrame,
        *,
        with_residual_ar: bool = True,
        residual_history: Any = None,
    ) -> pd.Series:
        """Generate point forecasts for evaluation dataset matching index and length."""
        components = self.predict_components(
            eval_df,
            with_residual_ar=with_residual_ar,
            residual_history=residual_history,
        )
        return pd.Series(components["total"], index=eval_df.index, dtype=float)

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
