"""Verification script for india_store.csv dataset preprocessing, feature governance, model training, and SHAP features."""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml_engine.preprocessing.intelligence import (
    classify_columns,
    analyze_forecasting_suitability,
    clean_and_impute_missing_values,
)
from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.explainability.forecast_explainer import ForecastExplainer


def audit_india_store():
    # Locate uploaded india_store.csv
    csv_file = ROOT / "storage" / "uploads" / "75284a3208ba496baf24a3a505302262.csv"
    if not csv_file.exists():
        csv_file = list((ROOT / "storage" / "uploads").glob("*.csv"))[0]

    raw_df = pd.read_csv(csv_file)
    print("==================================================")
    print(f"AUDITING DATASET: {csv_file.name}")
    print("==================================================")
    raw_cols = list(raw_df.columns)
    print(f"1. Raw Columns ({len(raw_cols)}):")
    print(raw_cols)

    # 2. Run Dynamic Preprocessing
    prep_res = prepare_dynamic_dataset(raw_df, domain="retail")
    cleaning_summary = prep_res["cleaning_summary"]

    print("\n2. Data Quality & Cleaning Summary:")
    print(f"   - Raw Rows:                       {cleaning_summary['raw_row_count']}")
    print(f"   - Exact Duplicates Removed:       {cleaning_summary['exact_duplicates_removed']}")
    print(f"   - Rows After Exact Dedup:         {cleaning_summary['cleaned_row_count_after_dedup']}")
    print(f"   - Missing Values Before:          {cleaning_summary['missing_values_before']}")
    print(f"   - Missing Values Handled:         {cleaning_summary['missing_values_handled']}")
    print(f"   - Invalid Rows Removed:           {cleaning_summary['rows_removed_due_to_invalid_data']}")
    print(f"   - Duplicate Timestamp Count:      {cleaning_summary['duplicate_timestamps_count']}")
    print(f"   - Rows Before Timestamp Agg:      {cleaning_summary['rows_before_timestamp_agg']}")
    print(f"   - Final Cleaned Rows:             {cleaning_summary['final_cleaned_rows']}")

    # 3. Excluded Columns
    excluded_cols = prep_res["cleaning_summary"]["excluded_features"]
    print(f"\n3. Excluded Columns ({len(excluded_cols)}):")
    print(excluded_cols)

    # 4. Load Materialized Feature Splits
    feature_dir = Path(prep_res["features_dir"])
    train_df = pd.read_parquet(feature_dir / "train.parquet")
    test_df = pd.read_parquet(feature_dir / "test.parquet")

    model_input_features = [c for c in train_df.columns if c not in ("timestamp", "sales")]
    print(f"\n4. Model Input Features ({len(model_input_features)}):")
    print(model_input_features)

    # 5. Fit Forecaster & SHAP Explainer
    rf = RandomForestForecaster(n_estimators=10)
    rf.fit(train_df, target_column="sales")

    explainer = ForecastExplainer(rf, model_name="random_forest")
    global_exp = explainer.explain_global(test_df)
    shap_features = [f["feature"] for f in global_exp["features"]]

    print(f"\n5. SHAP Features ({len(shap_features)}):")
    print(shap_features)

    # 6. Verification Assertions
    print("\n==================================================")
    print("VERIFICATION ASSERTIONS")
    print("==================================================")

    # Check Personal/ID exclusions
    forbidden = ["Customer ID", "Customer Name", "Last Name", "Date of Birth", "Postal Code", "Order ID", "Product ID", "Product Name"]
    for f in forbidden:
        assert f in excluded_cols, f"Error: '{f}' was NOT excluded!"
        assert f not in model_input_features, f"Error: '{f}' leaked into model input features!"
        assert f not in shap_features, f"Error: '{f}' leaked into SHAP features!"

    print("[SUCCESS] Personal/Identifier Exclusions Verified 100%! (0 forbidden features in model matrix)")

    # Check SHAP feature match
    assert set(shap_features) == set(model_input_features), f"Mismatch: SHAP features {shap_features} != Model features {model_input_features}"
    print("[SUCCESS] SHAP_FEATURES == FINAL_MODEL_FEATURES Verified 100%!")


if __name__ == "__main__":
    audit_india_store()

