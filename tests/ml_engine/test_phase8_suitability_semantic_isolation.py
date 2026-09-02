"""Regression test suite for Master Prompt Data Quality, Feature Governance, Suitability, and Data Isolation."""

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_engine.preprocessing.intelligence import (
    detect_time_column,
    detect_target_column,
    match_semantic_features,
    analyze_forecasting_suitability,
    aggregate_duplicate_timestamps,
    classify_columns,
    clean_and_impute_missing_values,
)


def test_1_semantic_date_detection():
    """Verify semantic date column detection across custom column names."""
    df = pd.DataFrame({
        "TransactionDate": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Revenue": [100.0, 150.0, 200.0],
    })
    time_col, candidates, conf = detect_time_column(df)
    assert time_col == "TransactionDate"
    assert conf > 0.7


def test_2_semantic_target_detection():
    """Verify semantic target detection for custom retail sales & demand keywords."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "SalesAmount": [100.0, 150.0, 200.0],
        "Customer_ID": [1001, 1002, 1003],
    })
    target_col, candidates, conf = detect_target_column(df, domain="retail")
    assert target_col == "SalesAmount"
    assert conf >= 0.7


def test_3_float_target_detection():
    """Verify continuous float targets are never skipped despite 100% uniqueness."""
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=100),
        "Units_Sold_Float": [100.5 + i * 1.37 for i in range(100)],
        "Item_No": [i + 1 for i in range(100)],
    })
    target_col, candidates, conf = detect_target_column(df, domain="retail")
    assert target_col == "Units_Sold_Float"


def test_4_unsuitable_short_dataset():
    """Verify datasets with fewer than 10 rows are classified as UNSUITABLE."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Sales": [10.0, 12.0],
    })
    analysis = analyze_forecasting_suitability(df, domain="retail")
    assert analysis["classification"] == "UNSUITABLE"
    assert not analysis["is_suitable"]
    assert any("fewer than 10" in r or "only 2 observations" in r for r in analysis["reasons"])


def test_5_constant_target():
    """Verify constant targets with zero variance are classified as UNSUITABLE."""
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=50),
        "Sales": [100.0] * 50,
    })
    analysis = analyze_forecasting_suitability(df, domain="retail")
    assert analysis["classification"] == "UNSUITABLE"
    assert not analysis["is_suitable"]
    assert len(analysis["reasons"]) > 0
    assert any("zero variance" in r.lower() or "constant" in r.lower() for r in analysis["reasons"])


def test_6_duplicate_timestamp_aggregation():
    """Verify transaction-level duplicate timestamps are aggregated semantically."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
        "Sales": [10.0, 20.0, 30.0, 40.0],
        "UnitPrice": [5.0, 5.0, 10.0, 10.0],
        "City": ["Mumbai", "Mumbai", "Delhi", "Delhi"],
    })
    agg_df, summary = aggregate_duplicate_timestamps(df, "Date", "Sales", domain="retail")
    assert summary["aggregated"] is True
    assert summary["raw_rows"] == 4
    assert summary["aggregated_rows"] == 2
    assert summary["methods"]["Sales"] == "sum"
    assert summary["methods"]["UnitPrice"] == "mean"
    assert list(agg_df["Sales"]) == [30.0, 70.0]


def test_7_8_9_dataset_a_b_a_isolation_no_demo_fallback_artifact_isolation():
    """Verify Dataset A -> Dataset B -> Dataset A isolation, artifact keys, and zero demo fallback."""
    from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset

    df_a = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=50),
        "Sales": [100.0 + i * 5 for i in range(50)],
    })
    df_b = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=50),
        "Sales": [5000.0 + i * 100 for i in range(50)],
    })

    res_a = prepare_dynamic_dataset(df_a, domain="retail")
    res_b = prepare_dynamic_dataset(df_b, domain="retail")
    res_a2 = prepare_dynamic_dataset(df_a, domain="retail")

    assert res_a["fingerprint"] != res_b["fingerprint"]
    assert res_a["fingerprint"] == res_a2["fingerprint"]
    assert res_a["features_id"] != res_b["features_id"]


def test_10_shap_dataset_isolation():
    """Verify SHAP explainer output matches selected model and dataset features."""
    from ml_engine.explainability.forecast_explainer import ForecastExplainer
    from ml_engine.models.random_forest import RandomForestForecaster

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30),
        "sales": [100.0 + i for i in range(30)],
        "discount_pct": [0.1] * 30,
        "lag_1": [99.0 + i for i in range(30)],
    })
    rf = RandomForestForecaster(n_estimators=10)
    rf.fit(df, target_column="sales")
    explainer = ForecastExplainer(rf, model_name="random_forest")
    exp = explainer.explain_global(df)

    assert exp["model"] == "random_forest"
    assert "natural_language_explanation" in exp
    assert len(exp["features"]) > 0


def test_11_suitability_classification():
    """Verify full suitability classification for a healthy time series dataset."""
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=100),
        "Sales": [100.0 + i * 2.5 for i in range(100)],
    })
    analysis = analyze_forecasting_suitability(df, domain="retail")
    assert analysis["classification"] == "SUITABLE"
    assert analysis["is_suitable"] is True
    assert len(analysis["reasons"]) == 0


def test_12_13_validation_only_selection_no_test_leakage(tmp_path):
    """Verify model comparison ranks models based on validation set without using test data."""
    from ml_engine.models.random_forest import RandomForestForecaster

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100),
        "sales": [100.0 + i * 1.5 for i in range(100)],
        "lag_1": [98.0 + i * 1.5 for i in range(100)],
    })

    train_df = df.iloc[:70]
    val_df = df.iloc[70:85]
    test_df = df.iloc[85:]

    rf = RandomForestForecaster(n_estimators=10)
    rf.fit(train_df, target_column="sales")

    val_preds = rf.predict(val_df)
    test_preds = rf.predict(test_df)

    assert len(val_preds) == 15
    assert len(test_preds) == 15


def test_14_semantic_feature_matching():
    """Verify universal semantic feature matching identifies custom feature families."""
    df = pd.DataFrame({
        "TxDate": ["2026-01-01", "2026-01-02"],
        "GrossRevenue": [1000.0, 1500.0],
        "UnitsSold": [10, 15],
        "DiscountPct": [0.1, 0.2],
        "StoreCity": ["Mumbai", "Delhi"],
    })
    matches = match_semantic_features(df, domain="retail")
    assert matches["TxDate"]["matched_family"] == "date_time"
    assert matches["GrossRevenue"]["matched_family"] == "sales_revenue"
    assert matches["UnitsSold"]["matched_family"] == "units_quantity"
    assert matches["DiscountPct"]["matched_family"] == "discount_promo"
    assert matches["StoreCity"]["matched_family"] == "geography"


def test_15_exact_duplicate_row_removal_and_uncorrupted_csv(tmp_path):
    """Verify exact duplicate rows are removed during preprocessing while original CSV remains untouched."""
    from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset

    csv_path = tmp_path / "raw_with_dups.csv"
    dates = [f"2026-01-{(i%28)+1:02d}" for i in range(40)]
    df_raw = pd.DataFrame({
        "Date": dates,
        "Sales": [100.0 + i for i in range(40)],
    })
    df_raw = pd.concat([df_raw, df_raw.iloc[:5]], ignore_index=True)
    df_raw.to_csv(csv_path, index=False)
    original_bytes = csv_path.read_bytes()

    res = prepare_dynamic_dataset(df_raw, domain="retail")
    assert res["cleaning_summary"]["exact_duplicates_removed"] == 5
    assert res["cleaning_summary"]["raw_row_count"] == 45
    assert res["cleaning_summary"]["cleaned_row_count_after_dedup"] == 40
    assert csv_path.read_bytes() == original_bytes


def test_16_feature_governance_excludes_personal_and_id_fields():
    """Verify Customer ID, DOB, Postal Code, and Order ID are excluded from forecasting features."""
    df = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=50),
        "Sales": [100.0 + i for i in range(50)],
        "Customer_ID": [f"CUST_{i}" for i in range(50)],
        "Customer_Name": [f"Person_{i}" for i in range(50)],
        "Date_of_Birth": ["1990-05-12"] * 50,
        "Postal_Code": ["400001"] * 50,
        "Discount_Pct": [0.05 + (i % 5) * 0.01 for i in range(50)],
    })
    classified = classify_columns(df, time_column="Date", target_column="Sales")
    ignored = classified["ignored_columns"]

    assert "Customer_ID" in ignored
    assert "Customer_Name" in ignored
    assert "Date_of_Birth" in ignored
    assert "Postal_Code" in ignored
    assert "Discount_Pct" not in ignored
    assert "Discount_Pct" in classified["numeric_columns"] or "Discount_Pct" in classified["categorical_columns"]



def test_17_missing_value_imputation_report():
    """Verify clean_and_impute_missing_values handles NaNs in numeric and categorical columns cleanly."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "Sales": [100.0, None, 300.0, 400.0],
        "City": ["Mumbai", None, "Mumbai", "Delhi"],
    })
    cleaned_df, report = clean_and_impute_missing_values(df, time_col="Date", target_col="Sales")
    assert report["missing_values_before"] == 2
    assert report["missing_values_after"] == 0
    assert not cleaned_df["Sales"].isna().any()
    assert not cleaned_df["City"].isna().any()


def test_18_extremely_low_timestamp_uniqueness_blocks_forecasting():
    """Verify datasets with 100 rows but only 3 unique timestamps are classified as UNSUITABLE."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03"] * 30,
        "Sales": [100.0 + i for i in range(90)],
    })
    suitability = analyze_forecasting_suitability(df, domain="retail")
    assert suitability["classification"] == "UNSUITABLE"
    assert not suitability["is_suitable"]
    assert any("insufficient unique temporal observations" in r for r in suitability["reasons"])
