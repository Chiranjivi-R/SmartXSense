"""Orchestration and entry point for baseline, statistical, ML, deep-learning, and hybrid model training and evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ml_engine.evaluation.metrics import evaluate_forecast

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
    # Hybrid-specific parameters
    components: list[str] | None = None,
    hybrid_weighting: str = "inverse_rmse",
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
        }

        if save_artifact:
            artifacts_dir = Path(artifacts_root) / domain
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            artifact_filename = f"{domain}_hybrid_{uuid.uuid4().hex[:8]}.pkl"
            artifact_path = artifacts_dir / artifact_filename
            artifact_data = {
                "result": result,
                "weights": weights,
                "components": hybrid_forecaster.components,
            }
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact_data, f)
            result["artifact_path"] = str(artifact_path.as_posix())

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
    }

    if model_key in ("lstm", "cnn_lstm"):
        result["training_sequences"] = res["training_sequences"]
        result["validation_sequences"] = res["validation_sequences"]
        result["test_sequences"] = res["test_sequences"]
        result["epochs_trained"] = res["epochs_trained"]

    if feature_importances is not None:
        result["feature_importances"] = feature_importances

    if save_artifact:
        artifacts_dir = Path(artifacts_root) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_filename = f"{domain}_{model_key}_{uuid.uuid4().hex[:8]}.pkl"
        artifact_path = artifacts_dir / artifact_filename
        artifact_data = {
            "result": result,
            "model": res["model"],
        }
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact_data, f)
        result["artifact_path"] = str(artifact_path.as_posix())

    return result


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
        save_artifact=args.save_artifact,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
