"""Unit tests for adaptive model selector with mandatory test-leakage immunity regression test."""

import pytest

from ml_engine.evaluation.adaptive_selector import AdaptiveModelSelector, AdaptiveSelectorError


def test_adaptive_selector_validation_only_selection():
    reports = [
        {"model": "arima", "validation_metrics": {"rmse": 100.0}},
        {"model": "random_forest", "validation_metrics": {"rmse": 45.0}},
        {"model": "xgboost", "validation_metrics": {"rmse": 60.0}},
    ]

    selector = AdaptiveModelSelector(selection_metric="rmse")
    res = selector.select_best_model(reports)

    assert res["selected_model"] == "random_forest"
    assert res["selection_metric"] == "rmse"
    assert res["selection_split"] == "validation"
    assert "random_forest" in res["reason"]


def test_adaptive_selector_test_metrics_leakage_immunity():
    """MANDATORY REGRESSION TEST:

    Altering TEST set metrics dramatically while keeping VALIDATION set metrics
    identical MUST NOT alter the adaptively selected model.
    """
    reports_run_1 = [
        {
            "model": "model_a",
            "validation_metrics": {"rmse": 12.0, "mae": 10.0},
            "test_metrics": {"rmse": 500.0, "mae": 400.0},  # Terrible test metric
        },
        {
            "model": "model_b",
            "validation_metrics": {"rmse": 8.0, "mae": 6.0},
            "test_metrics": {"rmse": 1.0, "mae": 0.5},    # Outstanding test metric
        },
    ]

    reports_run_2 = [
        {
            "model": "model_a",
            "validation_metrics": {"rmse": 12.0, "mae": 10.0},
            "test_metrics": {"rmse": 0.01, "mae": 0.001},  # Now outstanding test metric
        },
        {
            "model": "model_b",
            "validation_metrics": {"rmse": 8.0, "mae": 6.0},
            "test_metrics": {"rmse": 9999.0, "mae": 8888.0}, # Now terrible test metric
        },
    ]

    selector = AdaptiveModelSelector(selection_metric="rmse")

    selection_1 = selector.select_best_model(reports_run_1)
    selection_2 = selector.select_best_model(reports_run_2)

    # Selected model MUST remain identical despite extreme test metric mutation
    assert selection_1["selected_model"] == "model_b"
    assert selection_2["selected_model"] == "model_b"
    assert selection_1["selected_model"] == selection_2["selected_model"]


def test_adaptive_selector_deterministic_tie_breaking():
    reports = [
        {"model": "model_first", "validation_metrics": {"rmse": 10.0}},
        {"model": "model_second", "validation_metrics": {"rmse": 10.0}},
    ]

    selector = AdaptiveModelSelector(selection_metric="rmse")
    res = selector.select_best_model(reports)

    # Input order preserved on exact tie
    assert res["selected_model"] == "model_first"


def test_adaptive_selector_invalid_metric():
    selector = AdaptiveModelSelector(selection_metric="invalid")
    with pytest.raises(AdaptiveSelectorError, match="Adaptive model selection failed"):
        selector.select_best_model([{"model": "m1", "validation_metrics": {"rmse": 10.0}}])
