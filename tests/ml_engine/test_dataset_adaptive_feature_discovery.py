"""Unit and engine tests for dataset-adaptive feature discovery, schema adaptation, and model applicability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.applicability import (
    assess_model_applicability,
    profile_dataset_for_applicability,
)
from ml_engine.models.capability_matrix import (
    DEFAULT_COMPARE_MODELS,
    build_model_capability_matrix,
    get_models_catalog_payload,
)
from ml_engine.preprocessing.feature_discovery import (
    adapt_frame,
    discover_features_and_resolve_schema,
    profile_columns,
)
from ml_engine.preprocessing.schema import normalise_domain_schema


def test_aep_hourly_electricity_discovery():
    """Scenario 1: AEP_hourly-style dataset (Datetime, AEP_MW) discovers timestamp and target."""
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "AEP_MW": 15000 + 3000 * np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 50, n),
    })

    resolved = discover_features_and_resolve_schema(df, domain="electricity")
    assert resolved.is_valid is True
    assert resolved.time_column == "Datetime"
    assert resolved.target_column == "AEP_MW"
    assert resolved.canonical_target == "demand_mw"
    assert resolved.frequency in ("hourly", "h")
    assert resolved.is_chronological is True

    # Test non-destructive adaptation
    adapted = adapt_frame(df, resolved, preserve_original=True)
    assert "timestamp" in adapted.columns
    assert "demand_mw" in adapted.columns
    assert "Datetime" in adapted.columns
    assert "AEP_MW" in adapted.columns

    # Test normalise_domain_schema
    norm = normalise_domain_schema(df, "electricity")
    assert "timestamp" in norm.columns
    assert "demand_mw" in norm.columns
    assert len(norm) == n


def test_canonical_schema_backward_compatibility():
    """Scenario 2: Canonical timestamp + demand_mw functions without regression."""
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "demand_mw": 2000.0 + np.random.normal(0, 20, n),
        "temperature": 25.0 + np.random.normal(0, 2, n),
    })

    resolved = discover_features_and_resolve_schema(df, domain="electricity")
    assert resolved.is_valid is True
    assert resolved.time_column == "timestamp"
    assert resolved.target_column == "demand_mw"
    assert resolved.canonical_target == "demand_mw"

    norm = normalise_domain_schema(df, "electricity")
    assert "timestamp" in norm.columns
    assert "demand_mw" in norm.columns
    assert "temperature" in norm.columns


def test_electricity_aliases_discovery():
    """Scenario 3: Various electricity aliases (grid_load_mw, Power, Load_kW) resolve correctly."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="h")

    # Variation A: TimeStamp and grid_load_mw
    df_a = pd.DataFrame({
        "TimeStamp": dates,
        "grid_load_mw": 5000 + np.random.normal(0, 10, n),
    })
    res_a = discover_features_and_resolve_schema(df_a, domain="electricity")
    assert res_a.target_column == "grid_load_mw"
    assert res_a.canonical_target == "demand_mw"

    # Variation B: date_time and Power
    df_b = pd.DataFrame({
        "date_time": dates,
        "Power": 1200 + np.random.normal(0, 10, n),
    })
    res_b = discover_features_and_resolve_schema(df_b, domain="electricity")
    assert res_b.target_column == "Power"
    assert res_b.canonical_target == "demand_mw"

    # Variation C: record_time and Load_kW
    df_c = pd.DataFrame({
        "record_time": dates,
        "Load_kW": 800 + np.random.normal(0, 5, n),
    })
    res_c = discover_features_and_resolve_schema(df_c, domain="electricity")
    assert res_c.target_column == "Load_kW"
    assert res_c.canonical_target == "demand_mw"


def test_retail_datasets_discovery():
    """Scenario 4: Retail datasets with diverse timestamp/target names resolve correctly."""
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Invoice_Date": dates.strftime("%d/%m/%Y"),
        "Units_Sold": np.random.randint(50, 500, n),
        "Store_ID": ["STR_01"] * n,
        "Selling_Price": np.random.uniform(10, 50, n),
    })

    resolved = discover_features_and_resolve_schema(df, domain="retail")
    assert resolved.is_valid is True
    assert resolved.time_column == "Invoice_Date"
    assert resolved.target_column == "Units_Sold"
    assert resolved.canonical_target == "sales"

    norm = normalise_domain_schema(df, "retail")
    assert "timestamp" in norm.columns
    assert "sales" in norm.columns
    assert "store_id" in norm.columns or "Store_ID" in norm.columns


def test_ambiguous_targets_detection():
    """Scenario 5: Multiple candidate targets prompt ambiguity unless explicit target is given."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Sales": 1000 + np.random.normal(0, 50, n),
        "Quantity": np.random.randint(10, 100, n),
        "Store": ["A"] * n,
    })

    # Without explicit target: ambiguous target detected
    resolved_ambiguous = discover_features_and_resolve_schema(df, domain="retail")
    assert resolved_ambiguous.is_ambiguous_target is True
    assert "Sales" in resolved_ambiguous.ambiguous_targets
    assert "Quantity" in resolved_ambiguous.ambiguous_targets

    # With explicit target: ambiguity resolved cleanly
    resolved_explicit = discover_features_and_resolve_schema(df, domain="retail", explicit_target="Quantity")
    assert resolved_explicit.is_ambiguous_target is False
    assert resolved_explicit.target_column == "Quantity"


def test_explicit_user_mappings_precedence():
    """Scenario 7: Explicit user selections strictly override heuristic detection."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Custom_Time": dates,
        "metric_a": np.random.uniform(10, 50, n),
        "metric_b": np.random.uniform(100, 500, n),
    })

    resolved = discover_features_and_resolve_schema(
        df,
        domain="retail",
        explicit_target="metric_b",
        explicit_time_col="Custom_Time",
        explicit_mapping={"metric_b": "sales", "Custom_Time": "timestamp"},
    )
    assert resolved.target_column == "metric_b"
    assert resolved.time_column == "Custom_Time"
    assert resolved.canonical_target == "sales"

    norm = normalise_domain_schema(df, "retail", target="metric_b", time_column="Custom_Time")
    assert "sales" in norm.columns
    assert "timestamp" in norm.columns


def test_invalid_datasets_detection():
    """Scenario 6: Zero-variance target and missing timestamps produce clear errors."""
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    # Case A: Constant target (zero variance)
    df_const = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": [100.0] * n,
    })
    res_const = discover_features_and_resolve_schema(df_const, domain="electricity")
    assert res_const.is_valid is False
    assert any("constant" in err.lower() or "variance" in err.lower() for err in res_const.errors)

    # Case B: Missing timestamps
    df_no_time = pd.DataFrame({
        "store": ["S1"] * n,
        "sales": np.random.normal(100, 10, n),
    })
    res_no_time = discover_features_and_resolve_schema(df_no_time, domain="retail")
    assert res_no_time.is_valid is False
    assert any("timestamp" in err.lower() or "time" in err.lower() for err in res_no_time.errors)


def test_dataset_profile_and_applicability_with_arbitrary_names():
    """Scenario 10: Raw datasets with arbitrary names profile correctly for model applicability."""
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Datetime": dates,
        "AEP_MW": 15000 + 2000 * np.sin(np.linspace(0, 10, n)),
    })

    prof = profile_dataset_for_applicability(df, domain="electricity")
    assert prof.has_timestamp is True
    assert prof.has_target is True
    assert prof.target_name == "AEP_MW"
    assert prof.target_numeric is True
    assert prof.is_constant_target is False
    assert prof.detected_frequency == "h"

    app = assess_model_applicability(prof, domain="electricity")
    assert app["models"]["naive"]["eligible"] is True
    assert app["models"]["seasonal_naive"]["eligible"] is True
    assert app["models"]["arima"]["eligible"] is True
    assert app["models"]["sarima"]["eligible"] is True


def test_sarima_in_electricity_capability_matrix():
    """Scenario 10 & 11: SARIMA and ARIMA are present in electricity default comparison candidate set."""
    assert "arima" in DEFAULT_COMPARE_MODELS["electricity"]
    assert "sarima" in DEFAULT_COMPARE_MODELS["electricity"]
    assert "trend_xgboost" in DEFAULT_COMPARE_MODELS["electricity"]

    catalog = get_models_catalog_payload()
    elec_defaults = catalog["default_compare_models"]["electricity"]
    assert "sarima" in elec_defaults
    assert "arima" in elec_defaults
