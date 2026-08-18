"""Unit tests for dataset profiling and intelligent schema inspection."""

import pandas as pd
import pytest

from ml_engine.preprocessing.intelligence import (
    classify_columns,
    detect_geographic_columns,
    detect_target_column,
    detect_time_column,
    profile_dataset,
)


def test_date_column_detection():
    df = pd.DataFrame({
        "timestamp_str": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "sales": [100, 200, 300],
    })
    selected, candidates, confidence = detect_time_column(df)
    assert selected == "timestamp_str"
    assert "timestamp_str" in candidates
    assert confidence > 0.5


def test_retail_target_detection():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "units_sold": [50, 60],
        "store_id": ["S01", "S02"],
    })
    selected, candidates, confidence = detect_target_column(df, domain="retail")
    assert selected == "units_sold"
    assert "units_sold" in candidates
    assert confidence >= 0.8


def test_electricity_target_detection():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00"],
        "demand_mw": [1200.5, 1310.2],
        "temperature_c": [22.4, 21.8],
    })
    selected, candidates, confidence = detect_target_column(df, domain="electricity")
    assert selected == "demand_mw"
    assert "demand_mw" in candidates
    assert confidence >= 0.8


def test_numerical_vs_categorical_classification():
    df = pd.DataFrame({
        "date": ["2026-01-01"] * 5,
        "target": [10, 20, 30, 40, 50],
        "price": [1.5, 2.5, 3.5, 4.5, 5.5],
        "category": ["Food", "Food", "Tech", "Tech", "Food"],
    })
    classes = classify_columns(df, time_column="date", target_column="target")
    assert "price" in classes["numeric_columns"]
    assert "category" in classes["categorical_columns"]


def test_missing_and_duplicate_detection():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-03"],
        "sales": [10, 10, None],
        "empty_col": [None, None, None],
    })
    profile = profile_dataset(df, domain="retail")
    assert profile["duplicate_rows"] == 1
    assert profile["missing_summary"] == {"sales": 1, "empty_col": 3}
    assert "empty_col" in profile["ignored_columns"]


def test_geographic_column_detection():
    df = pd.DataFrame({
        "date": ["2026-01-01"],
        "sales": [100],
        "city": ["Mumbai"],
        "state": ["Maharashtra"],
        "country": ["India"],
    })
    geo_cols = detect_geographic_columns(df)
    assert set(geo_cols) == {"city", "state", "country"}


def test_unsuitable_dataset_response():
    # Missing date and missing target
    df = pd.DataFrame({
        "user_name": ["Alice", "Bob"],
        "comment": ["Great product", "Fast shipping"],
    })
    profile = profile_dataset(df, domain="retail")
    assert profile["suitable"] is False
    assert profile["confidence_label"] == "Low"
    assert len(profile["issues"]) >= 2
    assert "Dataset may not be fully suitable" in profile["explanation"]


def test_ambiguous_multiple_target_candidates():
    dates = [f"2026-01-{i:02d}" for i in range(1, 13)]
    df = pd.DataFrame({
        "date": dates,
        "units_sold": [10 + i for i in range(12)],
        "sales_revenue": [100.0 + i * 10 for i in range(12)],
    })
    profile = profile_dataset(df, domain="retail")
    assert profile["suitable"] is True
    assert len(profile["target_candidates"]) >= 2
    assert "units_sold" in profile["target_candidates"]
    assert "sales_revenue" in profile["target_candidates"]

    # Request explicit target override
    override_profile = profile_dataset(df, domain="retail", requested_target="sales_revenue")
    assert override_profile["selected_target"] == "sales_revenue"
