"""Orchestration and entry point for baseline, statistical, and ML model training and evaluation."""

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

    configuration: dict[str, Any] = {}
    feature_importances: dict[str, float] | None = None
    feature_names: list[str] = []

    if model_key == "arima":
        if domain == "retail":
            raise TrainerError("ARIMA model is currently supported only for the electricity domain")
        configuration["order"] = list(order)
        arima_model = ARIMAForecaster(order=order)
        arima_model.fit(train_df, target_column=target_column)
        model: Any = arima_model

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

    else:
        raise TrainerError(
            f"Unsupported model '{model_name}'. Supported models: naive, seasonal_naive, arima, baseline, random_forest, xgboost"
        )

    # Verify predictions alignment
    if len(val_pred) != len(val_df):
        raise TrainerError(f"Validation prediction length mismatch: got {len(val_pred)}, expected {len(val_df)}")
    if len(test_pred) != len(test_df):
        raise TrainerError(f"Test prediction length mismatch: got {len(test_pred)}, expected {len(test_df)}")

    val_metrics = evaluate_forecast(val_df[target_column], val_pred)
    test_metrics = evaluate_forecast(test_df[target_column], test_pred)

    result: dict[str, Any] = {
        "model": model_key,
        "domain": domain,
        "target": target_column,
        "configuration": configuration,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    if feature_importances is not None:
        result["feature_importances"] = feature_importances

    if save_artifact:
        artifacts_dir = Path(artifacts_root) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_filename = f"{domain}_{model_key}_{uuid.uuid4().hex[:8]}.pkl"
        artifact_path = artifacts_dir / artifact_filename
        artifact_data = {
            "result": result,
            "model": model,
        }
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact_data, f)
        result["artifact_path"] = str(artifact_path.as_posix())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models on Phase 5C feature artifacts.")
    parser.add_argument("--domain", choices=("retail", "electricity"), required=True, help="Dataset domain")
    parser.add_argument("--features-dir", type=Path, required=True, help="Path to Phase 5C feature directory containing train/validation/test parquets")
    parser.add_argument("--model", choices=("naive", "seasonal_naive", "arima", "baseline", "random_forest", "xgboost"), default="naive", help="Model family to train")
    parser.add_argument("--order", type=int, nargs=3, default=[1, 1, 1], metavar=("P", "D", "Q"), help="ARIMA order (p d q), default: 1 1 1")
    parser.add_argument("--seasonal-period", type=int, default=None, help="Seasonal period for seasonal naive model")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees for Random Forest / XGBoost")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth for Random Forest / XGBoost")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for XGBoost")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
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
        save_artifact=args.save_artifact,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
