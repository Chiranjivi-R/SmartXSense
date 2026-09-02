"""Tests for forecasting evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.evaluation.metrics import (
    EvaluationError,
    evaluate_forecast,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def test_mean_absolute_error_correctness() -> None:
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    # abs diffs: 2.0, 2.0, 3.0 -> mean = 7/3 = 2.3333333333333335
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(7.0 / 3.0)


def test_root_mean_squared_error_correctness() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([13.0, 16.0, 30.0])
    # sq diffs: 9.0, 16.0, 0.0 -> mean = 25/3 = 8.3333... -> sqrt = 2.88675...
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(25.0 / 3.0))


def test_mean_absolute_percentage_error_correctness() -> None:
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([110.0, 180.0])
    # % diffs: 10/100 = 0.10, 20/200 = 0.10 -> mean = 0.10
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(0.10)


def test_mape_zero_actual_handling() -> None:
    # Contains a zero target value
    y_true = [0.0, 100.0, 200.0]
    y_pred = [5.0, 110.0, 180.0]
    # Zero target ignored; non-zero targets percentage error = mean(0.10, 0.10) = 0.10
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert not np.isinf(mape)
    assert not np.isnan(mape)
    assert mape == pytest.approx(0.10)

    # All target values are zero
    y_true_all_zero = [0.0, 0.0, 0.0]
    y_pred_all_zero = [1.0, 2.0, 3.0]
    mape_all_zero = mean_absolute_percentage_error(y_true_all_zero, y_pred_all_zero)
    assert mape_all_zero == 0.0


def test_r2_score_correctness() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0, 50.0]
    y_pred = [10.0, 20.0, 30.0, 40.0, 50.0]
    # Perfect fit
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)

    # Imperfect fit
    y_pred_imperfect = [12.0, 18.0, 31.0, 39.0, 50.0]
    assert r2_score(y_true, y_pred_imperfect) < 1.0


def test_r2_score_zero_variance() -> None:
    y_true = [10.0, 10.0, 10.0]
    y_pred = [10.0, 10.0, 10.0]
    assert r2_score(y_true, y_pred) == pytest.approx(1.0)


def test_evaluate_forecast_report() -> None:
    y_true = [10.0, 20.0, 30.0]
    y_pred = [11.0, 22.0, 29.0]
    report = evaluate_forecast(y_true, y_pred)

    assert set(report.keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}

    for key, val in report.items():
        assert isinstance(val, float)
        assert not np.isnan(val)
        assert not np.isinf(val)


def test_invalid_input_length_mismatch() -> None:
    with pytest.raises(EvaluationError, match="Input length mismatch"):
        evaluate_forecast([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_input_handling() -> None:
    with pytest.raises(EvaluationError, match="cannot be empty"):
        evaluate_forecast([], [])


def test_nan_or_inf_input_handling() -> None:
    with pytest.raises(EvaluationError, match="contains NaN or infinite values"):
        evaluate_forecast([1.0, 2.0], [1.0, np.inf])


def test_metrics_consistency_unscaled_original_units() -> None:
    """Verify MAE, RMSE, MAPE, and R2 are calculated consistently on original target units."""
    # Actuals around 15-290, Predictions around 35-140
    y_true = np.array([200.0, 150.0, 250.0, 100.0, 290.0, 15.0])
    y_pred = np.array([80.0,  100.0, 110.0, 70.0,  140.0, 35.0])

    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)
    mape_val = mean_absolute_percentage_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    # MAE is average absolute error in original units (e.g. ~101.67)
    assert mae_val > 50.0
    # RMSE > MAE
    assert rmse_val >= mae_val
    # MAPE ratio is sum(|y-y_hat|/y)/N. For (120/200=0.6, 50/150=0.33, 140/250=0.56, 30/100=0.3, 150/290=0.517, 20/15=1.33), average ratio ~ 0.60
    assert 0.3 < mape_val < 2.0
    # R2 score is less than 1.0 for poor predictions
    assert r2_val < 1.0
