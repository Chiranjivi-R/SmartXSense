"""Unit tests for model comparison and ranking layer."""

import pytest

from ml_engine.evaluation.model_comparison import ModelComparisonError, compare_models


def test_compare_models_rmse_ranking():
    reports = [
        {
            "model": "arima",
            "validation_metrics": {"rmse": 100.0, "mae": 80.0, "mape": 0.1, "r2": 0.5},
            "test_metrics": {"rmse": 50.0},
        },
        {
            "model": "random_forest",
            "validation_metrics": {"rmse": 80.0, "mae": 60.0, "mape": 0.08, "r2": 0.7},
            "test_metrics": {"rmse": 120.0},
        },
        {
            "model": "hybrid",
            "validation_metrics": {"rmse": 60.0, "mae": 40.0, "mape": 0.05, "r2": 0.85},
            "test_metrics": {"rmse": 200.0},
        },
    ]

    res = compare_models(reports, selection_metric="rmse")

    assert res["selection_metric"] == "rmse"
    assert res["selection_split"] == "validation"
    assert res["recommended_model"] == "hybrid"
    assert len(res["ranking"]) == 3
    assert res["ranking"][0]["model"] == "hybrid"
    assert res["ranking"][0]["rank"] == 1
    assert res["ranking"][1]["model"] == "random_forest"
    assert res["ranking"][2]["model"] == "arima"


def test_compare_models_r2_ranking():
    reports = [
        {"model": "m1", "validation_metrics": {"r2": 0.6}},
        {"model": "m2", "validation_metrics": {"r2": 0.9}},
        {"model": "m3", "validation_metrics": {"r2": 0.75}},
    ]

    res = compare_models(reports, selection_metric="r2")
    assert res["recommended_model"] == "m2"
    assert res["ranking"][0]["model"] == "m2"
    assert res["ranking"][1]["model"] == "m3"
    assert res["ranking"][2]["model"] == "m1"


def test_compare_models_mae_and_mape():
    reports = [
        {"model": "m1", "validation_metrics": {"mae": 15.0, "mape": 0.15}},
        {"model": "m2", "validation_metrics": {"mae": 5.0, "mape": 0.05}},
    ]

    mae_res = compare_models(reports, selection_metric="mae")
    assert mae_res["recommended_model"] == "m2"

    mape_res = compare_models(reports, selection_metric="mape")
    assert mape_res["recommended_model"] == "m2"


def test_compare_models_invalid_inputs():
    with pytest.raises(ModelComparisonError, match="cannot be empty"):
        compare_models([])

    with pytest.raises(ModelComparisonError, match="Unsupported selection metric"):
        compare_models([{"model": "m1", "validation_metrics": {"rmse": 10.0}}], selection_metric="invalid_metric")

    with pytest.raises(ModelComparisonError, match="missing 'model' field"):
        compare_models([{"validation_metrics": {"rmse": 10.0}}])
