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
    derive_features_id,
    is_persistence_supported,
    predict_with_fitted_model,
    save_model_artifact,
)


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
from .hybrid import HybridForecaster, HybridModelError
from .lstm import DEFAULT_LOOKBACK, LSTMForecaster
from .random_forest import RandomForestForecaster
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
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
    lookback: int | None = None,
    lstm_units: int = 64,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 32,
    lstm_learning_rate: float = 0.001,
    cnn_filters: int = 32,
    kernel_size: int = 3,
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

    if model_key == "arima":
        if domain == "retail":
            raise TrainerError("ARIMA model is currently supported only for the electricity domain")
        configuration["order"] = list(order)
        arima_model = ARIMAForecaster(order=order)
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
        configuration.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        })
        rf_model = RandomForestForecaster(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        rf_model.fit(train_df, target_column=target_column)
        model = rf_model

        val_pred = rf_model.predict(val_df)
        test_pred = rf_model.predict(test_df)
        feature_importances = rf_model.get_feature_importances()
        feature_names = rf_model.feature_names

    elif model_key == "xgboost":
        depth = max_depth if max_depth is not None else 6
        configuration.update({
            "n_estimators": n_estimators,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        })
        xgb_model = XGBoostForecaster(
            n_estimators=n_estimators,
            max_depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        xgb_model.fit(train_df, target_column=target_column)
        model = xgb_model

        val_pred = xgb_model.predict(val_df)
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
            "naive, seasonal_naive, arima, baseline, random_forest, xgboost, lstm, cnn_lstm, hybrid"
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
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
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
    components: list[str] | None = None,
    hybrid_weighting: str = "inverse_rmse",
    prediction_limit: int = 100,
    save_artifact: bool = False,
    artifacts_root: Path | str = Path("ml_engine/artifacts/models"),
) -> dict[str, Any]:
    """Execute model training on train split and evaluation on validation and test splits."""
    domain = domain.lower().strip()
    if domain not in DOMAIN_TARGET_MAP:
        raise TrainerError(f"Unsupported domain '{domain}'. Supported domains: {list(DOMAIN_TARGET_MAP.keys())}")

    target_column = DOMAIN_TARGET_MAP[domain]
    features_dir = Path(features_dir)

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

    model_key = model_name.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    if model_key == "hybrid":
        try:
            hybrid_forecaster = HybridForecaster(
                components=components,
                domain=domain,
                weighting_strategy=hybrid_weighting,
            )
        except HybridModelError as err:
            raise TrainerError(f"Invalid hybrid configuration: {err}") from err

        val_predictions: dict[str, pd.Series] = {}
        test_predictions: dict[str, pd.Series] = {}
        component_val_metrics: dict[str, dict[str, float]] = {}
        component_test_metrics: dict[str, dict[str, float]] = {}

        for comp in hybrid_forecaster.components:
            res = _train_and_predict_single_model(
                model_key=comp,
                domain=domain,
                target_column=target_column,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                order=order,
                seasonal_period=seasonal_period,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                lookback=lookback,
                lstm_units=lstm_units,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                lstm_learning_rate=lstm_learning_rate,
                cnn_filters=cnn_filters,
                kernel_size=kernel_size,
            )
            v_pred = res["val_pred"]
            t_pred = res["test_pred"]

            val_predictions[comp] = v_pred
            test_predictions[comp] = t_pred

            v_targets = val_df.loc[v_pred.index, target_column]
            t_targets = test_df.loc[t_pred.index, target_column]

            component_val_metrics[comp] = evaluate_forecast(v_targets, v_pred)
            component_test_metrics[comp] = evaluate_forecast(t_targets, t_pred)

        try:
            weights = HybridForecaster.calculate_weights(
                component_val_metrics,
                weighting_strategy=hybrid_weighting,
            )
            val_hybrid_pred = HybridForecaster.combine(val_predictions, weights)
            test_hybrid_pred = HybridForecaster.combine(test_predictions, weights)
        except HybridModelError as err:
            raise TrainerError(f"Hybrid ensemble calculation failed: {err}") from err

        val_targets = val_df.loc[val_hybrid_pred.index, target_column]
        test_targets = test_df.loc[test_hybrid_pred.index, target_column]

        val_metrics = evaluate_forecast(val_targets, val_hybrid_pred)
        test_metrics = evaluate_forecast(test_targets, test_hybrid_pred)

        # This choice intentionally receives only validation facts.  The
        # held-out test metrics above are reported for evaluation, never for
        # choosing whether the ensemble is better than a component.
        adaptive_selection = HybridForecaster.select_adaptive_winner(
            component_val_metrics,
            val_metrics,
        )
        selected_model = adaptive_selection["selected_model"]
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

        result: dict[str, Any] = {
            "model": "hybrid",
            "domain": domain,
            "target": target_column,
            "configuration": {
                "components": hybrid_forecaster.components,
                "weighting": hybrid_weighting,
            },
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
            "adaptive_selection": adaptive_selection,
            "predictions": predictions_list,
            "prediction_count": pred_count,
            "total_test_predictions": total_test_preds,
        }

        # Hybrid persistence is intentionally unsupported: component models are
        # not retained after training, so weights-only artifacts cannot be reused.
        if save_artifact:
            result["artifact_persistence"] = {
                "supported": False,
                "reason": "hybrid requires fitted component models which are not retained",
            }

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
        seasonal_period=seasonal_period,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        lookback=lookback,
        lstm_units=lstm_units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lstm_learning_rate=lstm_learning_rate,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
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
                features_id = derive_features_id(features_dir)
                saved = save_model_artifact(
                    model=res["model"],
                    domain=domain,
                    model_key=model_key,
                    target=target_column,
                    features_id=features_id,
                    feature_names=feature_names,
                    configuration=res["configuration"],
                    validation_metrics=val_metrics,
                    test_metrics=test_metrics,
                    artifacts_root=artifacts_root,
                )
                result["artifact_path"] = saved["artifact_dir"]
                result["artifact_key"] = saved["relative_key"]
                result["artifact_persistence"] = {"supported": True, "saved": True}
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
    seasonal_period: int | None = None,
    n_estimators: int = 200,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    random_state: int = 42,
    lookback: int | None = None,
    lstm_units: int = 64,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 32,
    lstm_learning_rate: float = 0.001,
    cnn_filters: int = 32,
    kernel_size: int = 3,
) -> dict[str, Any]:
    """Mirror trainer configuration keys so artifact fingerprints match training saves."""
    model_key = model_key.lower().strip()
    if model_key == "baseline":
        model_key = "naive"

    if model_key == "arima":
        return {"order": list(order)}
    if model_key == "naive":
        return {"seasonal_period": 1}
    if model_key == "seasonal_naive":
        period = seasonal_period if seasonal_period is not None else DEFAULT_SEASONAL_PERIODS[domain]
        return {"seasonal_period": period}
    if model_key == "random_forest":
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
    if model_key == "xgboost":
        depth = max_depth if max_depth is not None else 6
        return {
            "n_estimators": n_estimators,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
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
        choices=("naive", "seasonal_naive", "arima", "baseline", "random_forest", "xgboost", "lstm", "cnn_lstm", "hybrid"),
        default="naive",
        help="Model family to train",
    )
    parser.add_argument("--order", type=int, nargs=3, default=[1, 1, 1], metavar=("P", "D", "Q"), help="ARIMA order (p d q), default: 1 1 1")
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
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
