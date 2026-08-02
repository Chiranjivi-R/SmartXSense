import pandas as pd
import pytest

from ml_engine.features.electricity import build_electricity_features
from ml_engine.features.retail import build_retail_features
from ml_engine.features.pipeline import prepare_forecasting_data
from ml_engine.preprocessing.domain import preprocess_frame


def test_retail_history_is_grouped_shifted_and_excludes_current_revenue():
    timestamps = pd.date_range("2025-01-01", periods=8, freq="D")
    frame = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "store_id": ["A"] * 8 + ["B"] * 8,
        "product_id": ["P"] * 16,
        "sales": list(range(1, 9)) + list(range(101, 109)),
        "revenue": list(range(10, 18)) + list(range(1000, 1008)),
        "price": [5] * 16,
    })
    features, names, report = build_retail_features(frame)
    a = features[features.store_id == "A"].sort_values("timestamp").reset_index(drop=True)
    b = features[features.store_id == "B"].sort_values("timestamp").reset_index(drop=True)
    assert a.loc[1, "lag_1"] == 1
    assert a.loc[7, "rolling_mean_7"] == pytest.approx(4)
    assert pd.isna(b.loc[0, "lag_1"])  # Group B cannot borrow Group A's history.
    assert "revenue" not in features and report["excluded_columns"] == ["revenue"]
    assert {"lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_std_28"} <= set(names)


def test_retail_aggregate_fallback_sums_target_without_fabricating_series():
    frame = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-01", "2025-01-02"], "sales": [2, 3, 7], "revenue": [20, 30, 70]})
    features, _, report = build_retail_features(frame)
    assert report["aggregate_fallback"] is True
    assert features.sales.tolist() == [5, 7]
    assert features.loc[1, "lag_1"] == 5


def test_features_accept_phase_5b_normalized_real_source_concepts():
    retail_raw = pd.DataFrame({"Invoice Date": ["2025-01-01", "2025-01-02"], "Units": [4, 6], "Brand": ["B", "B"], "Store Format": ["Supermarket", "Supermarket"], "Selling Price": [10, 10], "Revenue": [40, 60]})
    retail_processed, _, _ = preprocess_frame(retail_raw, "retail")
    retail_features, _, _ = build_retail_features(retail_processed)
    assert {"timestamp", "sales", "store_id", "product_id", "price"} <= set(retail_features.columns)
    electricity_raw = pd.DataFrame({"Year": [2025, 2025], "Date": ["01-01 00:00", "01-01 01:00"], "Hourly Demand Met (in MW)": [100, 110]})
    electricity_processed, _, _ = preprocess_frame(electricity_raw, "electricity")
    electricity_features, _, _ = build_electricity_features(electricity_processed)
    assert electricity_features.demand_mw.tolist() == [100, 110]


def test_electricity_lags_rollings_duplicates_and_gaps_are_safe():
    timestamps = pd.date_range("2025-01-01", periods=170, freq="h").delete(50)
    frame = pd.DataFrame({"timestamp": list(timestamps) + [timestamps[0]], "demand_mw": list(range(len(timestamps))) + [9999]})
    features, names, report = build_electricity_features(frame)
    assert len(features) == len(timestamps)
    assert features.loc[1, "lag_1"] == 0
    assert features.loc[24, "lag_24"] == 0
    assert features.loc[168, "lag_168"] == 0
    assert features.loc[24, "rolling_mean_24"] == pytest.approx(sum(range(24)) / 24)
    assert report["duplicate_timestamps_removed"] == 1
    assert report["irregular_gaps"]
    assert {"lag_1", "lag_24", "lag_48", "lag_168", "rolling_mean_168", "rolling_std_24"} <= set(names)


def test_split_boundaries_can_use_earlier_history_but_never_same_or_future_targets():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="h"), "demand_mw": range(10)})
    prepared = prepare_forecasting_data(frame, "electricity", train_ratio=.6, validation_ratio=.2, test_ratio=.2)
    validation_first = prepared.split.validation.iloc[0]
    test_first = prepared.split.test.iloc[0]
    assert validation_first.lag_1 == 5  # last legitimate training target
    assert test_first.lag_1 == 7  # last legitimate validation target
    assert validation_first.lag_1 != validation_first.demand_mw


def test_one_row_per_timestamp_retail_dataset_uses_aggregate_fallback_and_populates_history():
    timestamps = pd.date_range("2022-01-01", periods=35, freq="D")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "sales": list(range(10, 45)),
        "product_id": [f"PROD_{i % 5}" for i in range(35)],
        "category": [f"CAT_{i % 2}" for i in range(35)],
        "city": [f"CITY_{i % 3}" for i in range(35)],
        "price": [100.0] * 35,
        "revenue": [val * 100.0 for val in range(10, 45)],
    })
    features, names, report = build_retail_features(frame)

    assert report["aggregate_fallback"] is True
    assert report["series_columns"] == []
    assert pd.isna(features.loc[0, "lag_1"])
    assert features.loc[1, "lag_1"] == 10
    assert pd.isna(features.loc[6, "lag_7"])
    assert features.loc[7, "lag_7"] == 10
    assert pd.isna(features.loc[13, "lag_14"])
    assert features.loc[14, "lag_14"] == 10
    assert pd.isna(features.loc[27, "lag_28"])
    assert features.loc[28, "lag_28"] == 10
    assert pd.isna(features.loc[6, "rolling_mean_7"])
    assert features.loc[7, "rolling_mean_7"] == pytest.approx(sum(range(10, 17)) / 7.0)
    assert features.loc[28, "rolling_mean_28"] == pytest.approx(sum(range(10, 38)) / 28.0)
    assert {"product_id", "category", "city", "price"} <= set(features.columns)
    assert "revenue" not in features.columns


def test_genuine_grouped_retail_dataset_isolates_group_histories():
    timestamps = pd.date_range("2025-01-01", periods=30, freq="D")
    frame = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "store_id": ["STORE_A"] * 30 + ["STORE_B"] * 30,
        "product_id": ["PROD_1"] * 60,
        "sales": list(range(1, 31)) + list(range(501, 531)),
    })
    features, names, report = build_retail_features(frame)

    assert report["aggregate_fallback"] is False
    assert "store_id" in report["series_columns"]

    store_a = features[features.store_id == "STORE_A"].sort_values("timestamp").reset_index(drop=True)
    store_b = features[features.store_id == "STORE_B"].sort_values("timestamp").reset_index(drop=True)

    assert pd.isna(store_b.loc[0, "lag_1"])
    assert store_a.loc[1, "lag_1"] == 1
    assert store_b.loc[1, "lag_1"] == 501
    assert store_a.loc[7, "rolling_mean_7"] == pytest.approx(sum(range(1, 8)) / 7.0)
    assert store_b.loc[7, "rolling_mean_7"] == pytest.approx(sum(range(501, 508)) / 7.0)


def test_rolling_features_use_only_prior_observations_without_leakage():
    timestamps = pd.date_range("2025-01-01", periods=10, freq="D")
    sales = [10, 20, 30, 40, 50, 60, 70, 8000, 90, 100]
    frame = pd.DataFrame({"timestamp": timestamps, "sales": sales})
    features, _, _ = build_retail_features(frame)

    expected_mean_at_7 = sum([10, 20, 30, 40, 50, 60, 70]) / 7.0
    assert features.loc[7, "rolling_mean_7"] == pytest.approx(expected_mean_at_7)
    assert features.loc[7, "rolling_mean_7"] != pytest.approx((sum([10, 20, 30, 40, 50, 60, 70]) + 8000) / 8.0)

    expected_mean_at_8 = sum([20, 30, 40, 50, 60, 70, 8000]) / 7.0
    assert features.loc[8, "rolling_mean_7"] == pytest.approx(expected_mean_at_8)

