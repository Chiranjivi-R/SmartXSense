"""Orchestration and entry point for baseline, statistical, ML, deep-learning, and hybrid model training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from ml_engine.evaluation.metrics import evaluate_forecast
from ml_engine.models.artifact_store import (
    ArtifactError,
    compute_dataset_fingerprint,
    derive_features_id,
    is_persistence_supported,
    predict_with_fitted_model,
    save_model_artifact,
    try_load_compatible_artifact,
)

_DEFAULT_SENTINEL = object()


def format_chart_predictions(
    test_df: pd.DataFrame,
    test_pred: pd.Series,
    target_column: str,
    prediction_limit: int = 100,
) -> tuple[list[dict[str, Any]], int, int]:
    """Format test predictions as chart-ready JSON structures with deterministic sampling if needed."""
    if test_pred is None or len(test_pred) == 0:
        return [], 0, 0

    aligned_test_df = test_df.loc[test_pred.index]
    timestamps = aligned_test_df["timestamp"]
    actuals = aligned_test_df[target_column]

    n_total = len(test_pred)

    if prediction_limit <= 0 or n_total <= prediction_limit:
        sample_positions = list(range(n_total))
    else:
        raw_positions = np.round(np.linspace(0, n_total - 1, num=prediction_limit)).astype(int)
        sample_positions = list(dict.fromkeys(raw_positions))

    predictions = []
    for pos in sample_positions:
        ts_val = timestamps.iloc[pos]
        if hasattr(ts_val, "isoformat"):
            ts_str = ts_val.isoformat()
        else:
            ts_str = str(ts_val)

        actual_val = float(actuals.iloc[pos])
        pred_val = float(test_pred.iloc[pos])

        predictions.append({
            "timestamp": ts_str,
            "actual": actual_val,
            "predicted": pred_val,
        })

    return predictions, len(predictions), n_total

from .arima import ARIMAForecaster
from .baseline import NaiveForecaster
from .cnn_lstm import CNNLSTMForecaster, CNNLSTMModelError, DEFAULT_LOOKBACK_CNN_LSTM
from .hybrid import DEFAULT_HYBRID_COMPONENTS, HybridForecaster, HybridModelError, generate_hybrid_candidates
from .lstm import DEFAULT_LOOKBACK, LSTMForecaster
from .random_forest import RETAIL_MACRO_EXCLUSIONS, RandomForestForecaster
from .xgboost_model import XGBoostForecaster


class TrainerError(ValueError):
    """Raised when model training or evaluation pipeline fails."""


DOMAIN_TARGET_MAP = {
    "electricity": "demand_mw",
    "retail": "sales",
}

DEFAULT_SEASONAL_PERIODS = {
    "electricity": 24,
    "retail": 7,
}


def _train_and_predict_single_model(
    *,
    model_key: str,
    domain: str,
    target_column: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    training_window: int | None = None,
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
    target_transform: str | None = None,
    sample_weight_strategy: Any = _DEFAULT_SENTINEL,
    lookback: int | None = None,
    lstm_units: int = 64,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 32,
    lstm_learning_rate: float = 0.001,
    cnn_filters: int = 32,
    kernel_size: int = 3,
    trend_mode: str | None = None,
    trend_knot_hour: float | None = None,
    residual_ar: bool | None = None,
    residual_ar_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Internal helper to train a single component model and generate validation and test predictions."""
    model_key = model_key.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    configuration: dict[str, Any] = {}
    feature_importances: dict[str, float] | None = None
    feature_names: list[str] = []
    training_sequences: int | None = None
    validation_sequences: int | None = None
    test_sequences: int | None = None
    epochs_trained: int | None = None
    model: Any = None

    if model_key in ("arima", "sarima"):
        from ml_engine.models.applicability import assess_model_applicability
        app_res = assess_model_applicability(train_df, target_column=target_column, domain=domain)["models"].get(model_key)
        if app_res and not app_res["eligible"]:
            raise TrainerError(f"Model '{model_key}' is not applicable: [{app_res['reason_code']}] {app_res['reason']}")
        effective_order = (0, 1, 1) if (model_key == "sarima" and order == (1, 1, 1) and seasonal_order is None) else order
        effective_seasonal = (0, 1, 1, 24) if (model_key == "sarima" and seasonal_order is None) else seasonal_order
        effective_window = 8760 if (model_key == "sarima" and training_window is None) else training_window

        configuration["order"] = list(effective_order)
        if effective_seasonal is not None:
            configuration["seasonal_order"] = list(effective_seasonal)
        if effective_window is not None:
            configuration["training_window"] = int(effective_window)

        arima_model = ARIMAForecaster(
            order=effective_order,
            seasonal_order=effective_seasonal,
            training_window=effective_window,
        )
        arima_model.fit(train_df, target_column=target_column)
        model = arima_model

        val_pred = arima_model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined_train_val = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = arima_model.predict(test_df, target_column=target_column, history=combined_train_val)

    elif model_key in ("naive", "seasonal_naive"):
        if model_key == "naive":
            period = 1
        else:
            period = seasonal_period if seasonal_period is not None else DEFAULT_SEASONAL_PERIODS[domain]
        configuration["seasonal_period"] = period
        naive_model = NaiveForecaster(seasonal_period=period)
        naive_model.fit(train_df, target_column=target_column)
        model = naive_model

        val_pred = naive_model.predict(val_df, target_column=target_column, history=train_df[target_column])
        combined_train_val = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        test_pred = naive_model.predict(test_df, target_column=target_column, history=combined_train_val)

    elif model_key == "random_forest":
        resolved_target_transform = target_transform if target_transform is not None else ("log1p" if domain == "retail" else None)
        resolved_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if domain == "retail" else None
        configuration.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        })
        if resolved_target_transform is not None:
            configuration["target_transform"] = resolved_target_transform
        if resolved_exclusions is not None:
            configuration["feature_exclusions"] = resolved_exclusions

        rf_model = RandomForestForecaster(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            target_transform=resolved_target_transform,
            feature_exclusions=resolved_exclusions,
        )
        rf_model.fit(train_df, target_column=target_column)
        model = rf_model

        val_pred = rf_model.predict(val_df)
        test_pred = rf_model.predict(test_df)
        feature_importances = rf_model.get_feature_importances()
        feature_names = rf_model.feature_names

    elif model_key in ("xgboost", "trend_xgboost"):
        if model_key == "trend_xgboost" and domain == "retail":
            raise TrainerError("Model 'trend_xgboost' is only supported for the electricity domain")
        if model_key == "trend_xgboost":
            from ml_engine.models.applicability import assess_model_applicability
            app_res = assess_model_applicability(train_df, target_column=target_column, domain=domain)["models"].get("trend_xgboost")
            if app_res and not app_res["eligible"]:
                raise TrainerError(f"Model 'trend_xgboost' is not applicable: [{app_res['reason_code']}] {app_res['reason']}")
            resolved_trend_mode = trend_mode if trend_mode in ("linear", "piecewise_linear") else "piecewise_linear"
        elif trend_mode in ("linear", "piecewise_linear"):
            resolved_trend_mode = trend_mode
        else:
            resolved_trend_mode = None

        if resolved_trend_mode in ("linear", "piecewise_linear") and domain == "retail":
            raise TrainerError(f"trend_mode='{resolved_trend_mode}' is only supported for the electricity domain")

        # Residual AR resolution: backward-compatible default is False
        resolved_residual_ar = bool(residual_ar) if residual_ar is not None else False

        if resolved_residual_ar:
            from ml_engine.models.applicability import profile_dataset_for_applicability
            p = profile_dataset_for_applicability(train_df, target_column=target_column, domain=domain)
            if not p.has_hourly_lag_structure:
                raise TrainerError("residual_ar requires canonical hourly lag structure (lag_1, lag_24, lag_168)")

        resolved_knot_hour = (
            float(trend_knot_hour)
            if trend_knot_hour is not None
            else (30000.0 if resolved_trend_mode == "piecewise_linear" else None)
        )
        resolved_target_transform = target_transform if target_transform is not None else ("log1p" if domain == "retail" else None)
        if sample_weight_strategy is _DEFAULT_SENTINEL:
            resolved_weight_strategy = "spike_sensitive" if (domain == "retail" or model_key == "trend_xgboost") else None
        else:
            resolved_weight_strategy = sample_weight_strategy
        default_depth = 3 if domain == "retail" else 6
        depth = max_depth if max_depth is not None else default_depth
        lr = learning_rate if learning_rate != 0.05 else (0.03 if domain == "retail" else 0.05)
        configuration.update({
            "n_estimators": n_estimators,
            "max_depth": depth,
            "learning_rate": lr,
            "random_state": random_state,
        })
        if resolved_target_transform is not None:
            configuration["target_transform"] = resolved_target_transform
        if resolved_weight_strategy is not None:
            configuration["sample_weight_strategy"] = resolved_weight_strategy
        if resolved_trend_mode is not None:
            configuration["trend_mode"] = resolved_trend_mode
            if resolved_trend_mode == "piecewise_linear":
                configuration["trend_knot_hour"] = resolved_knot_hour
        if resolved_residual_ar:
            configuration["residual_ar"] = True
            if residual_ar_weights is not None:
                configuration["residual_ar_weights"] = dict(residual_ar_weights)

        xgb_model = XGBoostForecaster(
            n_estimators=n_estimators,
            max_depth=depth,
            learning_rate=lr,
            random_state=random_state,
            target_transform=resolved_target_transform,
            sample_weight_strategy=resolved_weight_strategy,
            trend_mode=resolved_trend_mode,
            trend_knot_hour=resolved_knot_hour,
            residual_ar=resolved_residual_ar,
            residual_ar_weights=residual_ar_weights,
        )
        xgb_model.fit(train_df, target_column=target_column)
        model = xgb_model

        val_pred = xgb_model.predict(val_df)
        if xgb_model.residual_ar and target_column in val_df.columns:
            val_residuals = val_df[target_column].to_numpy(dtype=float) - val_pred.to_numpy(dtype=float)
            test_pred = xgb_model.predict(test_df, residual_history=val_residuals[-168:])
        else:
            test_pred = xgb_model.predict(test_df)
        feature_importances = xgb_model.get_feature_importances()
        feature_names = xgb_model.feature_names

    elif model_key == "lstm":
        effective_lookback = lookback if lookback is not None else DEFAULT_LOOKBACK.get(domain, 7)
        configuration.update({
            "lookback": effective_lookback,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lstm_learning_rate,
            "random_seed": random_state,
        })
        lstm_model = LSTMForecaster(
            lookback=effective_lookback,
            lstm_units=lstm_units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lstm_learning_rate,
            random_seed=random_state,
        )
        lstm_model.fit(train_df, target_column=target_column, val_df=val_df)
        model = lstm_model

        val_pred = lstm_model.predict(val_df, target_column=target_column, history_df=train_df)
        combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        test_pred = lstm_model.predict(test_df, target_column=target_column, history_df=combined_train_val_df)

        feature_names = lstm_model.feature_names
        epochs_trained = lstm_model.epochs_trained
        training_sequences = lstm_model.n_training_sequences
        validation_sequences = len(val_pred)
        test_sequences = len(test_pred)

    elif model_key == "cnn_lstm":
        effective_lookback = lookback if lookback is not None else DEFAULT_LOOKBACK_CNN_LSTM.get(domain, 7)
        configuration.update({
            "lookback": effective_lookback,
            "cnn_filters": cnn_filters,
            "kernel_size": kernel_size,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lstm_learning_rate,
            "random_seed": random_state,
        })
        cnn_lstm_model = CNNLSTMForecaster(
            lookback=effective_lookback,
            cnn_filters=cnn_filters,
            kernel_size=kernel_size,
            lstm_units=lstm_units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lstm_learning_rate,
            random_seed=random_state,
        )
        cnn_lstm_model.fit(train_df, target_column=target_column, val_df=val_df)
        model = cnn_lstm_model

        val_pred = cnn_lstm_model.predict(val_df, target_column=target_column, history_df=train_df)
        combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        test_pred = cnn_lstm_model.predict(test_df, target_column=target_column, history_df=combined_train_val_df)

        feature_names = cnn_lstm_model.feature_names
        epochs_trained = cnn_lstm_model.epochs_trained
        training_sequences = cnn_lstm_model.n_training_sequences
        validation_sequences = len(val_pred)
        test_sequences = len(test_pred)

    else:
        raise TrainerError(
            f"Unsupported model '{model_key}'. Supported models: "
            "naive, seasonal_naive, arima, baseline, random_forest, xgboost, trend_xgboost, lstm, cnn_lstm, hybrid"
        )

    return {
        "model_key": model_key,
        "model": model,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "configuration": configuration,
        "feature_importances": feature_importances,
        "feature_names": feature_names,
        "training_sequences": training_sequences,
        "validation_sequences": validation_sequences,
        "test_sequences": test_sequences,
        "epochs_trained": epochs_trained,
    }


def run_training_pipeline(
    *,
    domain: str,
    features_dir: Path | str,
    model_name: str,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    training_window: int | None = None,
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
    target_transform: str | None = None,
    sample_weight_strategy: Any = _DEFAULT_SENTINEL,
    # Deep-learning shared parameters (LSTM + CNN-LSTM)
    lookback: int | None = None,
    lstm_units: int = 64,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 32,
    lstm_learning_rate: float = 0.001,
    # CNN-LSTM-specific parameters
    cnn_filters: int = 32,
    kernel_size: int = 3,
    trend_mode: str | None = None,
    trend_knot_hour: float | None = None,
    residual_ar: bool | None = None,
    residual_ar_weights: dict[str, float] | None = None,
    components: list[str] | None = None,
    hybrid_weighting: str = "inverse_rmse",
    prediction_limit: int = 100,
    save_artifact: bool = False,
    force_retrain: bool = False,
    artifacts_root: Path | str = Path("ml_engine/artifacts/models"),
) -> dict[str, Any]:
    """Execute model training or load existing reusable artifact, returning predictions and evaluation metrics."""
    domain = domain.lower().strip()
    if domain not in DOMAIN_TARGET_MAP:
        raise TrainerError(f"Unsupported domain '{domain}'. Supported domains: {list(DOMAIN_TARGET_MAP.keys())}")

    target_column = DOMAIN_TARGET_MAP[domain]
    features_dir = Path(features_dir)

    model_key = model_name.lower().strip()
    if model_key == "baseline":
        model_key = "naive"
    if model_key == "trend_xgboost" and domain == "retail":
        raise TrainerError("Model 'trend_xgboost' is only supported for the electricity domain")
    if trend_mode in ("linear", "piecewise_linear") and domain == "retail":
        raise TrainerError(f"trend_mode='{trend_mode}' is only supported for the electricity domain")
    train_path = features_dir / "train.parquet"
    val_path = features_dir / "validation.parquet"
    test_path = features_dir / "test.parquet"

    for path, name in ((train_path, "train.parquet"), (val_path, "validation.parquet"), (test_path, "test.parquet")):
        if not path.exists():
            raise TrainerError(f"Required Phase 5C artifact '{name}' missing from directory: {features_dir}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    for df, name in ((train_df, "train"), (val_df, "validation"), (test_df, "test")):
        if target_column not in df.columns:
            raise TrainerError(f"Target column '{target_column}' missing from {name} split")
        if "timestamp" not in df.columns:
            raise TrainerError(f"Timestamp column 'timestamp' missing from {name} split")

    from ml_engine.models.applicability import assess_model_applicability, profile_dataset_for_applicability
    applicability_report = assess_model_applicability(train_df, target_column=target_column, domain=domain)
    model_app = applicability_report["models"].get(model_key)
    if model_app and not model_app["eligible"]:
        raise TrainerError(
            f"Model '{model_key}' is not applicable: [{model_app['reason_code']}] {model_app['reason']}"
        )
    if residual_ar:
        p = profile_dataset_for_applicability(train_df, target_column=target_column, domain=domain)
        if not p.has_hourly_lag_structure:
            raise TrainerError("residual_ar requires canonical hourly lag structure (lag_1, lag_24, lag_168)")

    model_key = model_name.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    features_id = derive_features_id(features_dir)
    dataset_fingerprint = compute_dataset_fingerprint(features_dir)

    expected_config = expected_configuration_for_request(
        model_key=model_key,
        domain=domain,
        order=order,
        seasonal_order=seasonal_order,
        training_window=training_window,
        seasonal_period=seasonal_period,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        target_transform=target_transform,
        sample_weight_strategy=sample_weight_strategy,
        lookback=lookback,
        lstm_units=lstm_units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lstm_learning_rate=lstm_learning_rate,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
        trend_mode=trend_mode,
        trend_knot_hour=trend_knot_hour,
        residual_ar=residual_ar,
        residual_ar_weights=residual_ar_weights,
        components=components,
        hybrid_weighting=hybrid_weighting,
    )

    from .tabular_preprocessing import TabularFeatureTransformer

    rf_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (model_key == "random_forest" and domain == "retail") else ()
    expected_feature_names = TabularFeatureTransformer(target_column=target_column, extra_exclude=rf_exclusions).fit(train_df).feature_names

    # Attempt to reuse existing artifact if force_retrain is False and persistence is supported
    if not force_retrain and is_persistence_supported(model_key):
        loaded = try_load_compatible_artifact(
            domain=domain,
            model_key=model_key,
            target=target_column,
            features_id=features_id,
            dataset_fingerprint=dataset_fingerprint,
            feature_names=expected_feature_names,
            configuration=expected_config,
            artifacts_root=artifacts_root,
        )
        if loaded is not None:
            try:
                res = evaluate_fitted_model_on_splits(
                    domain=domain,
                    model_key=model_key,
                    model=loaded["model"],
                    features_dir=features_dir,
                    configuration=expected_config,
                    prediction_limit=prediction_limit,
                    feature_names=loaded["metadata"].get("feature_names"),
                )
                res["artifact_path"] = loaded["artifact_dir"]
                res["artifact_key"] = loaded["relative_key"]
                res["artifact_persistence"] = {"supported": True, "saved": True, "reused": True}
                return res
            except Exception:
                # Corrupt or un-evaluable artifact -> fall through to retraining.
                pass

    if model_key == "hybrid":
        if components is not None:
            raw_pool = components
        else:
            raw_pool = DEFAULT_HYBRID_COMPONENTS.get(domain, ["random_forest", "xgboost"])

        try:
            pool_components = HybridForecaster.validate_components(raw_pool, domain)
            hybrid_forecaster = HybridForecaster(
                components=pool_components,
                domain=domain,
                weighting_strategy=hybrid_weighting,
            )
        except HybridModelError as err:
            raise TrainerError(f"Invalid hybrid configuration: {err}") from err

        val_predictions: dict[str, pd.Series] = {}
        test_predictions: dict[str, pd.Series] = {}
        component_val_metrics: dict[str, dict[str, float]] = {}
        component_test_metrics: dict[str, dict[str, float]] = {}
        component_models: dict[str, Any] = {}

        for comp in pool_components:
            comp_config = expected_configuration_for_request(
                model_key=comp,
                domain=domain,
                order=order,
                seasonal_order=seasonal_order,
                training_window=training_window,
                seasonal_period=seasonal_period,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                target_transform=target_transform,
                sample_weight_strategy=sample_weight_strategy,
                lookback=lookback,
                lstm_units=lstm_units,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                lstm_learning_rate=lstm_learning_rate,
                cnn_filters=cnn_filters,
                kernel_size=kernel_size,
            )
            comp_loaded = None
            if not force_retrain and is_persistence_supported(comp):
                comp_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (comp == "random_forest" and domain == "retail") else ()
                comp_expected_features = TabularFeatureTransformer(target_column=target_column, extra_exclude=comp_exclusions).fit(train_df).feature_names
                comp_loaded = try_load_compatible_artifact(
                    domain=domain,
                    model_key=comp,
                    target=target_column,
                    features_id=features_id,
                    dataset_fingerprint=dataset_fingerprint,
                    feature_names=comp_expected_features,
                    configuration=comp_config,
                    artifacts_root=artifacts_root,
                )
            if comp_loaded is not None:
                comp_model = comp_loaded["model"]
                comp_pred_res = predict_with_fitted_model(
                    model_key=comp,
                    model=comp_model,
                    domain=domain,
                    target_column=target_column,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
                v_pred = comp_pred_res["val_pred"]
                t_pred = comp_pred_res["test_pred"]
            else:
                res = _train_and_predict_single_model(
                    model_key=comp,
                    domain=domain,
                    target_column=target_column,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    order=order,
                    seasonal_order=seasonal_order,
                    training_window=training_window,
                    seasonal_period=seasonal_period,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    target_transform=target_transform,
                    sample_weight_strategy=sample_weight_strategy,
                    lookback=lookback,
                    lstm_units=lstm_units,
                    dropout=dropout,
                    epochs=epochs,
                    batch_size=batch_size,
                    lstm_learning_rate=lstm_learning_rate,
                    cnn_filters=cnn_filters,
                    kernel_size=kernel_size,
                    trend_mode=comp_config.get("trend_mode"),
                    trend_knot_hour=comp_config.get("trend_knot_hour"),
                    residual_ar=comp_config.get("residual_ar"),
                    residual_ar_weights=comp_config.get("residual_ar_weights"),
                )
                comp_model = res["model"]
                v_pred = res["val_pred"]
                t_pred = res["test_pred"]

                if save_artifact and is_persistence_supported(comp):
                    try:
                        v_targs = val_df.loc[v_pred.index, target_column]
                        t_targs = test_df.loc[t_pred.index, target_column]
                        save_model_artifact(
                            model=comp_model,
                            domain=domain,
                            model_key=comp,
                            target=target_column,
                            features_id=features_id,
                            dataset_fingerprint=dataset_fingerprint,
                            feature_names=res.get("feature_names") or [],
                            configuration=comp_config,
                            validation_metrics=evaluate_forecast(v_targs, v_pred),
                            test_metrics=evaluate_forecast(t_targs, t_pred),
                            artifacts_root=artifacts_root,
                        )
                    except Exception:
                        pass

            component_models[comp] = comp_model
            val_predictions[comp] = v_pred
            test_predictions[comp] = t_pred

            v_targets = val_df.loc[v_pred.index, target_column]
            t_targets = test_df.loc[t_pred.index, target_column]

            component_val_metrics[comp] = evaluate_forecast(v_targets, v_pred)
            component_test_metrics[comp] = evaluate_forecast(t_targets, t_pred)

        # Dynamic or explicit candidate generation
        if components is not None:
            candidate_list = [hybrid_forecaster.components]
        else:
            candidate_list = generate_hybrid_candidates(
                domain=domain,
                applicable_models=list(component_models.keys()),
            )

        try:
            evaluated_hybrids = HybridForecaster.evaluate_hybrid_candidates(
                candidates=candidate_list,
                component_val_predictions=val_predictions,
                val_targets=val_df[target_column],
                component_test_predictions=test_predictions,
                test_targets=test_df[target_column],
                domain=domain,
                weighting_strategy=hybrid_weighting,
                selection_metric="rmse",
            )
            decision = HybridForecaster.select_best_hybrid_and_standalone(
                evaluated_hybrids=evaluated_hybrids,
                component_val_metrics=component_val_metrics,
                selection_metric="rmse",
            )
        except HybridModelError as err:
            raise TrainerError(f"Hybrid candidate evaluation failed: {err}") from err

        best_hybrid_raw = decision.get("_best_hybrid_raw", decision["best_hybrid"])
        best_hybrid = decision["best_hybrid"]
        weights = best_hybrid["weights"]
        val_hybrid_pred = best_hybrid_raw.get("val_pred")
        test_hybrid_pred = best_hybrid_raw.get("test_pred")
        val_metrics = best_hybrid["validation_metrics"]
        test_metrics = best_hybrid["test_metrics"]

        hybrid_forecaster.components = best_hybrid["components"]
        hybrid_forecaster.weights = weights
        hybrid_forecaster.component_models = {m: component_models[m] for m in best_hybrid["components"]}
        hybrid_forecaster.candidate_evaluations = decision["evaluated_candidates"]
        hybrid_forecaster.best_hybrid = best_hybrid
        hybrid_forecaster.best_standalone = decision["best_standalone"]
        hybrid_forecaster.selection_decision = {k: v for k, v in decision.items() if not k.startswith("_")}

        selected_model = decision["selected_model"]
        if selected_model == "hybrid":
            selected_test_pred = test_hybrid_pred
            selected_test_metrics = test_metrics
        else:
            selected_test_pred = test_predictions[selected_model]
            selected_test_metrics = component_test_metrics[selected_model]

        predictions_list, pred_count, total_test_preds = format_chart_predictions(
            test_df=test_df,
            test_pred=selected_test_pred,
            target_column=target_column,
            prediction_limit=prediction_limit,
        )

        effective_config = dict(expected_config)
        effective_config["components"] = best_hybrid["components"]

        clean_decision = {k: v for k, v in decision.items() if not k.startswith("_")}

        result: dict[str, Any] = {
            "model": "hybrid",
            "domain": domain,
            "target": target_column,
            "configuration": effective_config,
            "weights": weights,
            "component_validation_metrics": component_val_metrics,
            "component_test_metrics": component_test_metrics,
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "selected_model": selected_model,
            "selected_test_metrics": selected_test_metrics,
            "adaptive_selection": clean_decision,
            "hybrid_candidate_evaluations": decision["evaluated_candidates"],
            "predictions": predictions_list,
            "prediction_count": pred_count,
            "total_test_predictions": total_test_preds,
        }

        if save_artifact:
            try:
                saved = save_model_artifact(
                    model=hybrid_forecaster,
                    domain=domain,
                    model_key="hybrid",
                    target=target_column,
                    features_id=features_id,
                    dataset_fingerprint=dataset_fingerprint,
                    feature_names=expected_feature_names,

                    configuration=expected_config,
                    validation_metrics=val_metrics,
                    test_metrics=test_metrics,
                    artifacts_root=artifacts_root,
                )
                result["artifact_path"] = saved["artifact_dir"]
                result["artifact_key"] = saved["relative_key"]
                result["artifact_persistence"] = {"supported": True, "saved": True, "reused": False}
            except ArtifactError as err:
                result["artifact_persistence"] = {"supported": True, "saved": False, "reason": str(err)}

        return result

    # Single-model pipeline execution
    res = _train_and_predict_single_model(
        model_key=model_key,
        domain=domain,
        target_column=target_column,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        order=order,
        seasonal_order=seasonal_order,
        training_window=training_window,
        seasonal_period=seasonal_period,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        target_transform=target_transform,
        sample_weight_strategy=sample_weight_strategy,
        lookback=lookback,
        lstm_units=lstm_units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lstm_learning_rate=lstm_learning_rate,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
        trend_mode=trend_mode,
        trend_knot_hour=trend_knot_hour,
        residual_ar=residual_ar,
        residual_ar_weights=residual_ar_weights,
    )

    val_pred = res["val_pred"]
    test_pred = res["test_pred"]
    feature_names = res["feature_names"]
    feature_importances = res["feature_importances"]

    if model_key in ("lstm", "cnn_lstm"):
        val_targets = val_df.loc[val_pred.index, target_column]
        test_targets = test_df.loc[test_pred.index, target_column]
    else:
        if len(val_pred) != len(val_df):
            raise TrainerError(f"Validation prediction length mismatch: got {len(val_pred)}, expected {len(val_df)}")
        if len(test_pred) != len(test_df):
            raise TrainerError(f"Test prediction length mismatch: got {len(test_pred)}, expected {len(test_df)}")
        val_targets = val_df[target_column]
        test_targets = test_df[target_column]

    val_metrics = evaluate_forecast(val_targets, val_pred)
    test_metrics = evaluate_forecast(test_targets, test_pred)

    # Naive Baseline Comparison Evaluation
    try:
        naive_model = NaiveForecaster(seasonal_period=1)
        naive_model.fit(train_df, target_column=target_column)
        n_val_pred = naive_model.predict(val_df, target_column=target_column, history=train_df[target_column])
        comb_hist = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        n_test_pred = naive_model.predict(test_df, target_column=target_column, history=comb_hist)
        n_val_targets = val_df.loc[n_val_pred.index, target_column] if model_key in ("lstm", "cnn_lstm") else val_df[target_column]
        n_test_targets = test_df.loc[n_test_pred.index, target_column] if model_key in ("lstm", "cnn_lstm") else test_df[target_column]
        baseline_test_metrics = evaluate_forecast(n_test_targets, n_test_pred)
    except Exception:
        baseline_test_metrics = {"rmse": test_metrics["rmse"], "mae": test_metrics["mae"], "r2": 0.0, "mape": 0.0, "wape": 0.0, "smape": 0.0}

    b_rmse = baseline_test_metrics["rmse"]
    m_rmse = test_metrics["rmse"]
    improvement_pct = round(float((b_rmse - m_rmse) / (abs(b_rmse) + 1e-8) * 100), 2)
    beats_baseline = (m_rmse < b_rmse) and (test_metrics["r2"] > 0)

    if test_metrics["r2"] < 0 or not beats_baseline:
        forecast_quality = "Needs Improvement"
    elif improvement_pct > 15.0:
        forecast_quality = "Good"
    else:
        forecast_quality = "Moderate"

    predictions_list, pred_count, total_test_preds = format_chart_predictions(
        test_df=test_df,
        test_pred=test_pred,
        target_column=target_column,
        prediction_limit=prediction_limit,
    )

    result = {
        "model": model_key,
        "domain": domain,
        "target": target_column,
        "configuration": res["configuration"],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_test_metrics,
        "improvement_pct": improvement_pct,
        "beats_baseline": beats_baseline,
        "forecast_quality": forecast_quality,
        "predictions": predictions_list,
        "prediction_count": pred_count,
        "total_test_predictions": total_test_preds,
    }


    if model_key in ("lstm", "cnn_lstm"):
        result["training_sequences"] = res["training_sequences"]
        result["validation_sequences"] = res["validation_sequences"]
        result["test_sequences"] = res["test_sequences"]
        result["epochs_trained"] = res["epochs_trained"]

    if feature_importances is not None:
        result["feature_importances"] = feature_importances

    if save_artifact:
        if is_persistence_supported(model_key):
            try:
                saved = save_model_artifact(
                    model=res["model"],
                    domain=domain,
                    model_key=model_key,
                    target=target_column,
                    features_id=features_id,
                    dataset_fingerprint=dataset_fingerprint,
                    feature_names=feature_names,
                    configuration=res["configuration"],
                    validation_metrics=val_metrics,
                    test_metrics=test_metrics,
                    artifacts_root=artifacts_root,
                )
                result["artifact_path"] = saved["artifact_dir"]
                result["artifact_key"] = saved["relative_key"]
                result["artifact_persistence"] = {"supported": True, "saved": True, "reused": False}
            except ArtifactError as err:
                result["artifact_persistence"] = {
                    "supported": True,
                    "saved": False,
                    "reason": str(err),
                }
        else:
            result["artifact_persistence"] = {
                "supported": False,
                "reason": f"model '{model_key}' does not support safe persistence",
            }

    return result


def evaluate_fitted_model_on_splits(
    *,
    domain: str,
    model_key: str,
    model: Any,
    features_dir: Path | str,
    configuration: dict[str, Any] | None = None,
    prediction_limit: int = 100,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate an already-fitted model on feature splits and return API-shaped results."""
    domain = domain.lower().strip()
    if domain not in DOMAIN_TARGET_MAP:
        raise TrainerError(f"Unsupported domain '{domain}'. Supported domains: {list(DOMAIN_TARGET_MAP.keys())}")

    model_key = model_key.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    target_column = DOMAIN_TARGET_MAP[domain]
    features_dir = Path(features_dir)
    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "validation.parquet")
    test_df = pd.read_parquet(features_dir / "test.parquet")
    from .tabular_preprocessing import TabularFeatureTransformer

    rf_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (model_key == "random_forest" and domain == "retail") else ()
    expected_feature_names = TabularFeatureTransformer(target_column=target_column, extra_exclude=rf_exclusions).fit(train_df).feature_names
    if hasattr(model, "feature_names") and model.feature_names:
        if list(model.feature_names) != list(expected_feature_names):
            raise TrainerError(
                f"Model feature schema mismatch: model expects {len(model.feature_names)} features, "
                f"dataset evaluation split has {len(expected_feature_names)} features"
            )

    try:
        pred_res = predict_with_fitted_model(
            model_key=model_key,
            model=model,
            domain=domain,
            target_column=target_column,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
    except Exception as err:
        raise TrainerError(f"Failed to predict with loaded artifact: {err}") from err

    val_pred = pred_res["val_pred"]
    test_pred = pred_res["test_pred"]
    resolved_feature_names = list(feature_names if feature_names is not None else pred_res["feature_names"])

    if model_key == "hybrid":
        val_predictions = pred_res["val_predictions"]
        test_predictions = pred_res["test_predictions"]
        component_val_metrics: dict[str, dict[str, float]] = {}
        component_test_metrics: dict[str, dict[str, float]] = {}
        for comp, vp in val_predictions.items():
            v_targets = val_df.loc[vp.index, target_column]
            component_val_metrics[comp] = evaluate_forecast(v_targets, vp)
        for comp, tp in test_predictions.items():
            t_targets = test_df.loc[tp.index, target_column]
            component_test_metrics[comp] = evaluate_forecast(t_targets, tp)

        val_targets = val_df.loc[val_pred.index, target_column]
        test_targets = test_df.loc[test_pred.index, target_column]

        val_metrics = evaluate_forecast(val_targets, val_pred)
        test_metrics = evaluate_forecast(test_targets, test_pred)

        adaptive_selection = HybridForecaster.select_adaptive_winner(
            component_val_metrics,
            val_metrics,
        )
        selected_model = adaptive_selection["selected_model"]
        if selected_model == "hybrid":
            selected_test_pred = test_pred
            selected_test_metrics = test_metrics
        else:
            selected_test_pred = test_predictions[selected_model]
            selected_test_metrics = component_test_metrics[selected_model]

        predictions_list, pred_count, total_test_preds = format_chart_predictions(
            test_df=test_df,
            test_pred=selected_test_pred,
            target_column=target_column,
            prediction_limit=prediction_limit,
        )

        return {
            "model": "hybrid",
            "domain": domain,
            "target": target_column,
            "configuration": configuration or {"components": getattr(model, "components", []), "weighting": getattr(model, "weighting_strategy", "inverse_rmse")},
            "weights": getattr(model, "weights", {}),
            "component_validation_metrics": component_val_metrics,
            "component_test_metrics": component_test_metrics,
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "selected_model": selected_model,
            "selected_test_metrics": selected_test_metrics,
            "adaptive_selection": adaptive_selection,
            "predictions": predictions_list,
            "prediction_count": pred_count,
            "total_test_predictions": total_test_preds,
        }

    if model_key in ("lstm", "cnn_lstm"):
        val_targets = val_df.loc[val_pred.index, target_column]
        test_targets = test_df.loc[test_pred.index, target_column]
    else:
        if len(val_pred) != len(val_df):
            raise TrainerError(f"Validation prediction length mismatch: got {len(val_pred)}, expected {len(val_df)}")
        if len(test_pred) != len(test_df):
            raise TrainerError(f"Test prediction length mismatch: got {len(test_pred)}, expected {len(test_df)}")
        val_targets = val_df[target_column]
        test_targets = test_df[target_column]

    val_metrics = evaluate_forecast(val_targets, val_pred)
    test_metrics = evaluate_forecast(test_targets, test_pred)

    # Calculate baseline metrics for loaded artifact
    try:
        naive_model = NaiveForecaster(seasonal_period=1)
        naive_model.fit(train_df, target_column=target_column)
        comb_hist = pd.concat([train_df[target_column], val_df[target_column]], ignore_index=True)
        n_test_pred = naive_model.predict(test_df, target_column=target_column, history=comb_hist)
        n_test_targets = test_df.loc[n_test_pred.index, target_column] if model_key in ("lstm", "cnn_lstm") else test_df[target_column]
        baseline_test_metrics = evaluate_forecast(n_test_targets, n_test_pred)
    except Exception:
        baseline_test_metrics = {"rmse": test_metrics["rmse"], "mae": test_metrics["mae"], "r2": 0.0, "mape": 0.0, "wape": 0.0, "smape": 0.0}

    b_rmse = baseline_test_metrics["rmse"]
    m_rmse = test_metrics["rmse"]
    improvement_pct = round(float((b_rmse - m_rmse) / (abs(b_rmse) + 1e-8) * 100), 2)
    beats_baseline = (m_rmse < b_rmse) and (test_metrics["r2"] > 0)

    if test_metrics["r2"] < 0 or not beats_baseline:
        forecast_quality = "Needs Improvement"
    elif improvement_pct > 15.0:
        forecast_quality = "Good"
    else:
        forecast_quality = "Moderate"

    predictions_list, pred_count, total_test_preds = format_chart_predictions(
        test_df=test_df,
        test_pred=test_pred,
        target_column=target_column,
        prediction_limit=prediction_limit,
    )

    result: dict[str, Any] = {
        "model": model_key,
        "domain": domain,
        "target": target_column,
        "configuration": configuration or {},
        "feature_count": len(resolved_feature_names),
        "feature_names": resolved_feature_names,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_test_metrics,
        "improvement_pct": improvement_pct,
        "beats_baseline": beats_baseline,
        "forecast_quality": forecast_quality,
        "predictions": predictions_list,
        "prediction_count": pred_count,
        "total_test_predictions": total_test_preds,
    }


    if model_key in ("lstm", "cnn_lstm"):
        result["training_sequences"] = pred_res["training_sequences"]
        result["validation_sequences"] = pred_res["validation_sequences"]
        result["test_sequences"] = pred_res["test_sequences"]
        result["epochs_trained"] = pred_res["epochs_trained"]

    if pred_res["feature_importances"] is not None:
        result["feature_importances"] = pred_res["feature_importances"]

    return result


def expected_configuration_for_request(
    *,
    model_key: str,
    domain: str,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
    training_window: int | None = None,
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
    target_transform: str | None = None,
    sample_weight_strategy: Any = _DEFAULT_SENTINEL,
    lookback: int | None = None,
    lstm_units: int = 64,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 32,
    lstm_learning_rate: float = 0.001,
    cnn_filters: int = 32,
    kernel_size: int = 3,
    trend_mode: str | None = None,
    trend_knot_hour: float | None = None,
    residual_ar: bool | None = None,
    residual_ar_weights: dict[str, float] | None = None,
    components: list[str] | None = None,
    hybrid_weighting: str = "inverse_rmse",
) -> dict[str, Any]:
    """Mirror trainer configuration keys so artifact fingerprints match training saves."""
    model_key = model_key.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    if model_key == "hybrid":
        comps = components if components is not None else DEFAULT_HYBRID_COMPONENTS.get(domain, ["random_forest", "xgboost"])
        comps = HybridForecaster.validate_components(comps, domain)
        return {
            "components": comps,
            "weighting": hybrid_weighting,
        }

    if model_key in ("arima", "sarima"):
        effective_order = (0, 1, 1) if (model_key == "sarima" and order == (1, 1, 1) and seasonal_order is None) else order
        effective_seasonal = (0, 1, 1, 24) if (model_key == "sarima" and seasonal_order is None) else seasonal_order
        effective_window = 8760 if (model_key == "sarima" and training_window is None) else training_window

        cfg: dict[str, Any] = {"order": list(effective_order)}
        if effective_seasonal is not None:
            cfg["seasonal_order"] = list(effective_seasonal)
        if effective_window is not None:
            cfg["training_window"] = int(effective_window)
        return cfg
    if model_key == "naive":
        return {"seasonal_period": 1}
    if model_key == "seasonal_naive":
        period = seasonal_period if seasonal_period is not None else DEFAULT_SEASONAL_PERIODS[domain]
        return {"seasonal_period": period}
    if model_key == "random_forest":
        resolved_target_transform = target_transform if target_transform is not None else ("log1p" if domain == "retail" else None)
        cfg = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        if resolved_target_transform is not None:
            cfg["target_transform"] = resolved_target_transform
        if domain == "retail":
            cfg["feature_exclusions"] = list(RETAIL_MACRO_EXCLUSIONS)
        return cfg
    if model_key in ("xgboost", "trend_xgboost"):
        if model_key == "trend_xgboost" and domain == "retail":
            raise TrainerError("Model 'trend_xgboost' is only supported for the electricity domain")
        if model_key == "trend_xgboost":
            resolved_trend_mode = trend_mode if trend_mode in ("linear", "piecewise_linear") else "piecewise_linear"
        elif trend_mode in ("linear", "piecewise_linear"):
            resolved_trend_mode = trend_mode
        else:
            resolved_trend_mode = None

        if resolved_trend_mode in ("linear", "piecewise_linear") and domain == "retail":
            raise TrainerError(f"trend_mode='{resolved_trend_mode}' is only supported for the electricity domain")

        # Residual AR resolution: backward-compatible default is False
        resolved_residual_ar = bool(residual_ar) if residual_ar is not None else False

        resolved_knot_hour = (
            float(trend_knot_hour)
            if trend_knot_hour is not None
            else (30000.0 if resolved_trend_mode == "piecewise_linear" else None)
        )
        resolved_target_transform = target_transform if target_transform is not None else ("log1p" if domain == "retail" else None)
        if sample_weight_strategy is _DEFAULT_SENTINEL:
            resolved_weight_strategy = "spike_sensitive" if (domain == "retail" or model_key == "trend_xgboost") else None
        else:
            resolved_weight_strategy = sample_weight_strategy
        default_depth = 3 if domain == "retail" else 6
        depth = max_depth if max_depth is not None else default_depth
        lr = learning_rate if learning_rate != 0.05 else (0.03 if domain == "retail" else 0.05)
        cfg = {
            "n_estimators": n_estimators,
            "max_depth": depth,
            "learning_rate": lr,
            "random_state": random_state,
        }
        if resolved_target_transform is not None:
            cfg["target_transform"] = resolved_target_transform
        if resolved_weight_strategy is not None:
            cfg["sample_weight_strategy"] = resolved_weight_strategy
        if resolved_trend_mode is not None:
            cfg["trend_mode"] = resolved_trend_mode
            if resolved_trend_mode == "piecewise_linear":
                cfg["trend_knot_hour"] = resolved_knot_hour
        if resolved_residual_ar:
            cfg["residual_ar"] = True
            if residual_ar_weights is not None:
                cfg["residual_ar_weights"] = dict(residual_ar_weights)
        return cfg
    if model_key == "lstm":
        effective_lookback = lookback if lookback is not None else DEFAULT_LOOKBACK.get(domain, 7)
        return {
            "lookback": effective_lookback,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lstm_learning_rate,
            "random_seed": random_state,
        }
    if model_key == "cnn_lstm":
        effective_lookback = lookback if lookback is not None else DEFAULT_LOOKBACK_CNN_LSTM.get(domain, 7)
        return {
            "lookback": effective_lookback,
            "cnn_filters": cnn_filters,
            "kernel_size": kernel_size,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lstm_learning_rate,
            "random_seed": random_state,
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models on Phase 5C feature artifacts.")
    parser.add_argument("--domain", choices=("retail", "electricity"), required=True, help="Dataset domain")
    parser.add_argument("--features-dir", type=Path, required=True, help="Path to Phase 5C feature directory containing train/validation/test parquets")
    parser.add_argument(
        "--model",
        choices=("naive", "seasonal_naive", "arima", "sarima", "baseline", "random_forest", "xgboost", "trend_xgboost", "lstm", "cnn_lstm", "hybrid"),
        default="naive",
        help="Model family to train",
    )
    parser.add_argument("--trend-mode", choices=("linear", None), default=None, help="Trend detrending mode for electricity XGBoost (default: None)")
    parser.add_argument("--order", type=int, nargs=3, default=[1, 1, 1], metavar=("P", "D", "Q"), help="ARIMA order (p d q), default: 1 1 1")
    parser.add_argument("--seasonal-order", type=int, nargs=4, default=None, metavar=("P", "D", "Q", "S"), help="SARIMA seasonal order (P D Q s), default: None")
    parser.add_argument("--training-window", type=int, default=None, help="Training window size for ARIMA/SARIMA (default: None, 8760 for sarima)")
    parser.add_argument("--seasonal-period", type=int, default=None, help="Seasonal period for seasonal naive model")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees for Random Forest / XGBoost")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth for Random Forest / XGBoost")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for XGBoost")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lookback", type=int, default=None, help="LSTM lookback window (default: 24 for electricity, 7 for retail)")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM hidden units (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout rate (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=20, help="LSTM training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="LSTM batch size (default: 32)")
    parser.add_argument("--lstm-learning-rate", type=float, default=0.001, help="LSTM/CNN-LSTM learning rate (default: 0.001)")
    parser.add_argument("--cnn-filters", type=int, default=32, help="CNN-LSTM number of Conv1D filters (default: 32)")
    parser.add_argument("--kernel-size", type=int, default=3, help="CNN-LSTM Conv1D kernel size (default: 3; must be <= lookback)")
    parser.add_argument("--components", type=str, nargs="+", default=None, help="Component models for hybrid ensemble")
    parser.add_argument("--hybrid-weighting", choices=("inverse_rmse", "equal"), default="inverse_rmse", help="Hybrid weighting strategy")
    parser.add_argument("--prediction-limit", type=int, default=100, help="Maximum number of test predictions to return for chart plotting")
    parser.add_argument("--save-artifact", action="store_true", help="Save model artifact to ml_engine/artifacts/models/")

    args = parser.parse_args()

    result = run_training_pipeline(
        domain=args.domain,
        features_dir=args.features_dir,
        model_name=args.model,
        order=tuple(args.order),
        seasonal_order=tuple(args.seasonal_order) if args.seasonal_order is not None else None,
        training_window=args.training_window,
        seasonal_period=args.seasonal_period,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        lookback=args.lookback,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_learning_rate=args.lstm_learning_rate,
        cnn_filters=args.cnn_filters,
        kernel_size=args.kernel_size,
        components=args.components,
        hybrid_weighting=args.hybrid_weighting,
        prediction_limit=args.prediction_limit,
        save_artifact=args.save_artifact,
        trend_mode=args.trend_mode,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
