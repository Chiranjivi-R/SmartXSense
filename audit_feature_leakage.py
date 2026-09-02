"""Feature Leakage & Availability Audit for india_store.csv"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml_engine.preprocessing.intelligence import (
    classify_columns,
    analyze_forecasting_suitability,
)
from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.explainability.forecast_explainer import ForecastExplainer


FEATURES_AUDIT_LIST = [
    "day",
    "day_of_week",
    "discount",
    "is_weekend",
    "month",
    "profit",
    "quantity",
    "quarter",
    "year",
    "outlet_type",
    "city_type",
    "category_of_goods",
    "region",
    "country",
    "segment",
    "order_date",
    "ship_date",
    "ship_mode",
    "state",
    "sub_category",
    "season",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]


def audit_features():
    csv_file = ROOT / "storage" / "uploads" / "75284a3208ba496baf24a3a505302262.csv"
    raw_df = pd.read_csv(csv_file)

    print("==================================================")
    print("FEATURE LEAKAGE & AVAILABILITY AUDIT TABLE")
    print("==================================================")

    prep_res = prepare_dynamic_dataset(raw_df, domain="retail")
    feature_dir = Path(prep_res["features_dir"])
    train_df = pd.read_parquet(feature_dir / "train.parquet")

    current_model_features = [c for c in train_df.columns if c not in ("timestamp", "sales")]

    print(f"Current Model Features ({len(current_model_features)}):")
    print(current_model_features)


if __name__ == "__main__":
    audit_features()
