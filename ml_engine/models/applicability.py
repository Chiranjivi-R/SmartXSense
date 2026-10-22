"""Authoritative Dataset Suitability and Model Applicability Engine.

SmartXSense Framework — Phase 7B-7 Dynamic Model Applicability.

Strict Architectural Principles:
1. Model applicability is strictly derived from measurable dataset characteristics.
2. Zero hard-coded domain restrictions (e.g. NEVER `if domain == "retail": arima = unavailable`).
3. Differentiates Training Applicability from Future Forecasting Horizon Capability.
4. Produces structured N/A reason contracts for every model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    """Measurable dataset characteristics extracted for model applicability evaluation."""
    row_count: int
    unique_timestamps: int
    duplicate_timestamps: int
    timestamp_uniqueness_ratio: float
    has_timestamp: bool
    is_chronological: bool
    detected_frequency: str
    has_target: bool
    target_name: str
    target_numeric: bool
    target_missing_count: int
    target_missing_ratio: float
    target_variance: float
    is_constant_target: bool
    history_length: int
    usable_feature_count: int
    entity_columns: list[str]
    is_interleaved_panel: bool
    interleaved_details: str | None
    has_hourly_lag_structure: bool
    has_daily_lag_structure: bool
    available_lags: list[str]


@dataclass
class ModelApplicabilityResult:
    """Structured applicability contract for a single forecasting model."""
    model: str
    eligible: bool
    status: str  # "available" or "unavailable"
    reason_code: str | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POTENTIAL_ENTITY_COLUMNS = (
    "product_id", "item_id", "sku", "product", "item",
    "store_id", "store", "store_code", "location",
    "city", "state", "province", "category", "brand",
)

CANONICAL_HOURLY_LAGS = ("lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168")
CANONICAL_DAILY_LAGS = ("lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14")


def profile_dataset_for_applicability(
    data: pd.DataFrame | Path | str,
    target_column: str | None = None,
    domain: str | None = None,
) -> DatasetProfile:
    """Analyze a DataFrame or features directory to extract the 18 measurable characteristics."""
    # 1. Resolve DataFrame
    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.is_dir():
            train_p = p / "train.parquet"
            if train_p.exists():
                df = pd.read_parquet(train_p)
            else:
                raise ValueError(f"Features directory {p} does not contain train.parquet")
        else:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be a pandas DataFrame or Path to dataset/features directory")

    row_count = len(df)

    # 2. Timestamp analysis
    time_col = "timestamp" if "timestamp" in df.columns else None
    resolved_target = target_column

    if (time_col is None or resolved_target is None) and row_count > 0:
        try:
            from ml_engine.preprocessing.feature_discovery import discover_features_and_resolve_schema
            discovered = discover_features_and_resolve_schema(df, domain=domain, explicit_target=target_column)
            if time_col is None and discovered.time_column and discovered.time_column in df.columns:
                time_col = discovered.time_column
            if resolved_target is None and discovered.target_column and discovered.target_column in df.columns:
                resolved_target = discovered.target_column
        except Exception:
            pass

    has_timestamp = bool(time_col and time_col in df.columns)
    unique_timestamps = 0
    dup_timestamps = 0
    uniqueness_ratio = 0.0
    is_chronological = False
    detected_freq = "unknown"

    if has_timestamp and row_count > 0:
        ts_parsed = pd.to_datetime(df[time_col], errors="coerce")
        valid_ts = ts_parsed.dropna()
        unique_timestamps = int(valid_ts.nunique())
        dup_timestamps = int(len(valid_ts) - unique_timestamps)
        uniqueness_ratio = round(float(unique_timestamps / row_count), 4) if row_count > 0 else 0.0
        is_chronological = bool(valid_ts.is_monotonic_increasing)

        if len(valid_ts) > 2:
            diffs = valid_ts.sort_values().diff().dropna()
            mode_diff = diffs.mode()
            if not mode_diff.empty:
                td = mode_diff.iloc[0]
                secs = td.total_seconds()
                if 3500 <= secs <= 3700:
                    detected_freq = "h"
                elif 86000 <= secs <= 87000:
                    detected_freq = "D"
                elif 600000 <= secs <= 610000:
                    detected_freq = "W"
                else:
                    detected_freq = str(td)
            else:
                detected_freq = "irregular"

    # 3. Target resolution and analysis
    has_target = False
    target_name = ""
    target_numeric = False
    target_missing_count = 0
    target_missing_ratio = 1.0
    target_variance = 0.0
    is_constant_target = True

    if not resolved_target:
        if "sales" in df.columns:
            resolved_target = "sales"
        elif "demand_mw" in df.columns:
            resolved_target = "demand_mw"
        elif domain == "retail" and "sales" in df.columns:
            resolved_target = "sales"
        elif domain == "electricity" and "demand_mw" in df.columns:
            resolved_target = "demand_mw"
        else:
            # Fallback to numeric columns
            num_cols = [c for c in df.select_dtypes(include="number").columns if c not in ("timestamp", "year", "month", "day", time_col)]
            if num_cols:
                resolved_target = num_cols[0]

    if resolved_target and resolved_target in df.columns:
        has_target = True
        target_name = resolved_target
        target_s = pd.to_numeric(df[resolved_target], errors="coerce")
        valid_target = target_s.dropna()
        target_missing_count = int(target_s.isna().sum())
        target_missing_ratio = float(target_missing_count / row_count) if row_count > 0 else 1.0
        target_numeric = len(valid_target) > 0

        if len(valid_target) > 1:
            target_variance = float(valid_target.var())
            is_constant_target = bool(target_variance < 1e-8 or math.isnan(target_variance))
        elif len(valid_target) == 1:
            is_constant_target = True
            target_variance = 0.0

    # 4. Usable feature count (excluding timestamp, target, revenue)
    excluded = {"timestamp", "revenue", target_name}
    if time_col:
        excluded.add(time_col)
    feature_cols = [c for c in df.columns if c not in excluded]
    usable_feature_count = len(feature_cols)

    # 5. Entity Panel Analysis
    found_entity_cols = [c for c in POTENTIAL_ENTITY_COLUMNS if c in df.columns]
    multi_entities: dict[str, int] = {}
    is_interleaved = False
    interleaved_details = None

    if found_entity_cols and row_count > 0:
        for c in found_entity_cols:
            nu = df[c].nunique()
            if nu > 1:
                multi_entities[c] = int(nu)

        if multi_entities:
            # Test if entities switch frequently across consecutive rows (interleaved panel)
            primary_col = list(multi_entities.keys())[0]
            entity_shifts = int((df[primary_col] != df[primary_col].shift(1)).sum())
            # If entities change in more than 20% of consecutive rows, it is an interleaved multi-entity panel
            if entity_shifts > 0.20 * row_count:
                is_interleaved = True
                details_list = [f"{k} ({v} unique)" for k, v in multi_entities.items()]
                interleaved_details = f"Interleaved multi-entity panel with {', '.join(details_list)}."

    # 6. Lag structure analysis
    available_lags = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]
    has_hourly = any(c in df.columns for c in CANONICAL_HOURLY_LAGS)
    has_daily = any(c in df.columns for c in CANONICAL_DAILY_LAGS)

    return DatasetProfile(
        row_count=row_count,
        unique_timestamps=unique_timestamps,
        duplicate_timestamps=dup_timestamps,
        timestamp_uniqueness_ratio=uniqueness_ratio,
        has_timestamp=has_timestamp,
        is_chronological=is_chronological,
        detected_frequency=detected_freq,
        has_target=has_target,
        target_name=target_name,
        target_numeric=target_numeric,
        target_missing_count=target_missing_count,
        target_missing_ratio=target_missing_ratio,
        target_variance=target_variance,
        is_constant_target=is_constant_target,
        history_length=row_count,
        usable_feature_count=usable_feature_count,
        entity_columns=found_entity_cols,
        is_interleaved_panel=is_interleaved,
        interleaved_details=interleaved_details,
        has_hourly_lag_structure=has_hourly,
        has_daily_lag_structure=has_daily,
        available_lags=available_lags,
    )


def assess_model_applicability(
    profile_or_data: DatasetProfile | pd.DataFrame | Path | str,
    target_column: str | None = None,
    domain: str | None = None,
    requested_horizon: int | None = None,
) -> dict[str, Any]:
    """Evaluate applicability for all 10 supported model families strictly on dataset suitability."""
    if isinstance(profile_or_data, DatasetProfile):
        profile = profile_or_data
    else:
        profile = profile_dataset_for_applicability(profile_or_data, target_column=target_column, domain=domain)

    results: dict[str, ModelApplicabilityResult] = {}

    # Common basic prerequisites
    min_rows = profile.row_count
    has_valid_target = profile.has_target and profile.target_numeric and profile.target_missing_ratio < 0.5
    has_variance = not profile.is_constant_target

    # 1. NAIVE
    if not has_valid_target:
        results["naive"] = ModelApplicabilityResult(
            model="naive",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="Target variable is missing, non-numeric, or exceeds 50% missingness.",
        )
    elif min_rows < 2:
        results["naive"] = ModelApplicabilityResult(
            model="naive",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_HISTORY",
            reason="Naive baseline requires at least 2 historical observations.",
        )
    else:
        results["naive"] = ModelApplicabilityResult(
            model="naive",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 2. SEASONAL NAIVE
    s_period = 24 if profile.detected_frequency == "h" else 7
    if not has_valid_target:
        results["seasonal_naive"] = ModelApplicabilityResult(
            model="seasonal_naive",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="Target variable is missing or invalid.",
        )
    elif min_rows <= s_period:
        results["seasonal_naive"] = ModelApplicabilityResult(
            model="seasonal_naive",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_SEASONAL_HISTORY",
            reason=f"Seasonal Naive requires history longer than seasonal cycle (got {min_rows} rows, requires > {s_period}).",
        )
    else:
        results["seasonal_naive"] = ModelApplicabilityResult(
            model="seasonal_naive",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 3. ARIMA
    if not has_valid_target:
        results["arima"] = ModelApplicabilityResult(
            model="arima",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="ARIMA requires a valid numeric target series.",
        )
    elif not has_variance:
        results["arima"] = ModelApplicabilityResult(
            model="arima",
            eligible=False,
            status="unavailable",
            reason_code="CONSTANT_TARGET",
            reason="ARIMA requires a target series with positive variance (constant target detected).",
        )
    elif profile.is_interleaved_panel:
        results["arima"] = ModelApplicabilityResult(
            model="arima",
            eligible=False,
            status="unavailable",
            reason_code="INTERLEAVED_PANEL",
            reason=f"Dataset contains an interleaved multi-entity panel ({profile.interleaved_details}). "
                   "Univariate autoregressive modeling (ARIMA) is not applicable across changing entities without prior single-entity filtering or volume aggregation.",
        )
    elif min_rows < 30:
        results["arima"] = ModelApplicabilityResult(
            model="arima",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_HISTORY",
            reason=f"ARIMA requires at least 30 observations for stable parameter estimation (got {min_rows}).",
        )
    else:
        results["arima"] = ModelApplicabilityResult(
            model="arima",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 4. SARIMA
    sarima_min_obs = max(35, 2 * s_period)
    if not results["arima"].eligible:
        results["sarima"] = ModelApplicabilityResult(
            model="sarima",
            eligible=False,
            status="unavailable",
            reason_code=results["arima"].reason_code,
            reason=results["arima"].reason,
        )
    elif min_rows < sarima_min_obs:
        results["sarima"] = ModelApplicabilityResult(
            model="sarima",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_SEASONAL_HISTORY",
            reason=f"SARIMA requires at least {sarima_min_obs} observations for seasonal decomposition (got {min_rows}).",
        )
    else:
        results["sarima"] = ModelApplicabilityResult(
            model="sarima",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 5. RANDOM FOREST
    if not has_valid_target:
        results["random_forest"] = ModelApplicabilityResult(
            model="random_forest",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="Random Forest requires a valid numeric target.",
        )
    elif min_rows < 15:
        results["random_forest"] = ModelApplicabilityResult(
            model="random_forest",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_HISTORY",
            reason=f"N/A — Eligible training observation criterion not met. {min_rows} usable training observations available; minimum 15 required for Random Forest (got {min_rows}).",
        )
    elif profile.usable_feature_count < 1:
        results["random_forest"] = ModelApplicabilityResult(
            model="random_forest",
            eligible=False,
            status="unavailable",
            reason_code="NO_USABLE_FEATURES",
            reason="Random Forest requires at least 1 usable predictor feature.",
        )
    else:
        results["random_forest"] = ModelApplicabilityResult(
            model="random_forest",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 6. XGBOOST
    if not has_valid_target:
        results["xgboost"] = ModelApplicabilityResult(
            model="xgboost",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="XGBoost requires a valid numeric target.",
        )
    elif min_rows < 15:
        results["xgboost"] = ModelApplicabilityResult(
            model="xgboost",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_HISTORY",
            reason=f"N/A — Eligible training observation criterion not met. {min_rows} usable training observations available; minimum 15 required for XGBoost (got {min_rows}).",
        )
    elif profile.usable_feature_count < 1:
        results["xgboost"] = ModelApplicabilityResult(
            model="xgboost",
            eligible=False,
            status="unavailable",
            reason_code="NO_USABLE_FEATURES",
            reason="XGBoost requires at least 1 usable predictor feature.",
        )
    else:
        results["xgboost"] = ModelApplicabilityResult(
            model="xgboost",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 7. TREND XGBOOST
    if not results["xgboost"].eligible:
        results["trend_xgboost"] = ModelApplicabilityResult(
            model="trend_xgboost",
            eligible=False,
            status="unavailable",
            reason_code=results["xgboost"].reason_code,
            reason=results["xgboost"].reason,
        )
    elif not profile.has_timestamp:
        results["trend_xgboost"] = ModelApplicabilityResult(
            model="trend_xgboost",
            eligible=False,
            status="unavailable",
            reason_code="MISSING_TIMESTAMP",
            reason="Trend XGBoost requires a continuous timestamp coordinate to construct trend decomposition.",
        )
    elif not profile.has_hourly_lag_structure:
        results["trend_xgboost"] = ModelApplicabilityResult(
            model="trend_xgboost",
            eligible=False,
            status="unavailable",
            reason_code="NO_HOURLY_LAG_STRUCTURE",
            reason="Trend XGBoost requires canonical hourly lag features (lag_24, lag_168) and continuous hourly resolution for piecewise trend detrending.",
        )
    else:
        results["trend_xgboost"] = ModelApplicabilityResult(
            model="trend_xgboost",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 8. LSTM
    lstm_lookback = 24 if profile.detected_frequency == "h" else 7
    if not has_valid_target:
        results["lstm"] = ModelApplicabilityResult(
            model="lstm",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="LSTM requires a valid numeric target.",
        )
    elif min_rows <= (lstm_lookback + 10):
        results["lstm"] = ModelApplicabilityResult(
            model="lstm",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_SEQUENCE_HISTORY",
            reason=f"N/A — Eligible sequence history criterion not met. LSTM requires sufficient sequential history for lookback window ({min_rows} usable training observations available; requires > {lstm_lookback + 10}, got {min_rows} rows).",
        )
    else:
        results["lstm"] = ModelApplicabilityResult(
            model="lstm",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 9. CNN-LSTM
    cnn_min_rows = lstm_lookback + 3 + 10  # lookback + kernel + margin
    if not has_valid_target:
        results["cnn_lstm"] = ModelApplicabilityResult(
            model="cnn_lstm",
            eligible=False,
            status="unavailable",
            reason_code="INVALID_TARGET",
            reason="CNN-LSTM requires a valid numeric target.",
        )
    elif min_rows <= cnn_min_rows:
        results["cnn_lstm"] = ModelApplicabilityResult(
            model="cnn_lstm",
            eligible=False,
            status="unavailable",
            reason_code="INSUFFICIENT_SEQUENCE_HISTORY",
            reason=f"N/A — Eligible sequence history criterion not met. CNN-LSTM requires history > lookback + kernel_size + 10 ({min_rows} usable training observations available; requires > {cnn_min_rows}, got {min_rows} rows).",
        )
    else:
        results["cnn_lstm"] = ModelApplicabilityResult(
            model="cnn_lstm",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    # 10. HYBRID
    # Compatible components pool: {random_forest, xgboost, arima, sarima}
    compatible_candidates = ["random_forest", "xgboost", "arima", "sarima"]
    eligible_components = [m for m in compatible_candidates if results[m].eligible]

    if len(eligible_components) < 2:
        results["hybrid"] = ModelApplicabilityResult(
            model="hybrid",
            eligible=False,
            status="unavailable",
            reason_code="FEWER_THAN_TWO_COMPONENTS",
            reason=f"N/A — Eligible component criterion not met. Hybrid ensemble requires at least 2 distinct eligible component models (only {eligible_components} eligible).",
        )
    else:
        results["hybrid"] = ModelApplicabilityResult(
            model="hybrid",
            eligible=True,
            status="available",
            reason_code=None,
            reason=None,
        )

    eligible_models = [k for k, v in results.items() if v.eligible]
    ineligible_models = {k: v.to_dict() for k, v in results.items() if not v.eligible}

    return {
        "dataset_profile": asdict(profile),
        "models": {k: v.to_dict() for k, v in results.items()},
        "eligible_models": eligible_models,
        "ineligible_models": ineligible_models,
        "eligible_count": len(eligible_models),
        "ineligible_count": len(ineligible_models),
    }


def assess_future_capability(
    model_key: str,
    domain: str | None = None,
    horizon: int = 24,
    dataset_applicable: bool = True,
) -> dict[str, Any]:
    """Evaluate future forecasting capability at requested horizon (separate from dataset applicability)."""
    model_key = model_key.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    # Horizon-capped models
    MAX_HORIZONS = {"sarima": 168}

    if not dataset_applicable:
        return {
            "model": model_key,
            "future_capable": False,
            "status": "unavailable",
            "reason_code": "DATASET_INAPPLICABLE",
            "reason": f"Model '{model_key}' is not applicable to the current dataset structure.",
        }

    if model_key in ("naive", "seasonal_naive"):
        return {
            "model": model_key,
            "future_capable": False,
            "status": "unavailable",
            "reason_code": "BASELINE_TEST_ONLY",
            "reason": f"Model '{model_key}' is a reference baseline and does not support future forecasting.",
        }

    if model_key in ("lstm", "cnn_lstm"):
        return {
            "model": model_key,
            "future_capable": False,
            "status": "unavailable",
            "reason_code": "SEQUENCE_LOOKBACK_REQUIRED",
            "reason": f"Future forecasting is not enabled for sequence model '{model_key}' in Phase B (requires projected lookback arrays).",
        }

    if model_key in MAX_HORIZONS and horizon > MAX_HORIZONS[model_key]:
        return {
            "model": model_key,
            "future_capable": False,
            "status": "unavailable",
            "reason_code": "HORIZON_EXCEEDED",
            "reason": f"Model '{model_key}' future forecasting is restricted to horizon <= {MAX_HORIZONS[model_key]} (requested {horizon}).",
        }

    return {
        "model": model_key,
        "future_capable": True,
        "status": "available",
        "reason_code": None,
        "reason": None,
    }
