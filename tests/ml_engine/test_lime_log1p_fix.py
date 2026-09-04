"""Regression tests proving LIME uses original target units for log1p models and un-transformed units for non-log1p models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability.lime_explainer import LimeTabularExplainer


class DummyTransformer:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_names].to_numpy()


class DummyRawModel:
    def __init__(self, fixed_log_pred: float = 2.0):
        self.fixed_log_pred = fixed_log_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.fixed_log_pred, dtype=float)


class StubForecaster:
    def __init__(self, target_transform: str | None = "log1p", fixed_log_pred: float = 2.0):
        self._model = DummyRawModel(fixed_log_pred=fixed_log_pred)
        self._transformer = DummyTransformer(["f1", "f2"])
        self.target_transform = target_transform

    def predict(self, eval_df: pd.DataFrame) -> pd.Series:
        raw_preds = self._model.predict(self._transformer.transform(eval_df))
        if self.target_transform == "log1p":
            preds = np.expm1(raw_preds)
            preds = np.clip(preds, a_min=0.0, a_max=None)
        else:
            preds = raw_preds
        return pd.Series(preds, index=eval_df.index)


def test_lime_log1p_model_uses_inverse_transformed_predictions() -> None:
    """Established that LIME prediction function inverse-transforms predictions for log1p models."""
    log_val = 2.0
    expected_original_val = np.expm1(log_val)  # exp(2) - 1 = ~6.389

    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0] * 10, "f2": [4.0, 5.0, 6.0] * 10})
    eval_df = pd.DataFrame({"f1": [1.5], "f2": [4.5]})

    forecaster = StubForecaster(target_transform="log1p", fixed_log_pred=log_val)
    explainer = LimeTabularExplainer(forecaster, train_df=train_df)

    res = explainer.explain_local(eval_df, sample_index=0)

    # 1. Returned prediction matches expm1(log_val)
    assert res["prediction"] == pytest.approx(expected_original_val)

    # 2. Verify intercept + attribution sum approximates expm1(log_val), NOT raw 2.0
    total_lime_prediction = res["intercept"] + sum(item["weight"] for item in res["feature_attributions"])
    assert total_lime_prediction == pytest.approx(expected_original_val, abs=1.0)
    assert total_lime_prediction != pytest.approx(log_val, abs=1.0)


def test_lime_non_log1p_model_uses_raw_predictions() -> None:
    """Verify that non-log1p models are NOT inverse-transformed by LIME."""
    raw_val = 50.0

    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0] * 10, "f2": [4.0, 5.0, 6.0] * 10})
    eval_df = pd.DataFrame({"f1": [1.5], "f2": [4.5]})

    forecaster = StubForecaster(target_transform=None, fixed_log_pred=raw_val)
    explainer = LimeTabularExplainer(forecaster, train_df=train_df)

    res = explainer.explain_local(eval_df, sample_index=0)

    assert res["prediction"] == pytest.approx(raw_val)
    total_lime_prediction = res["intercept"] + sum(item["weight"] for item in res["feature_attributions"])
    assert total_lime_prediction == pytest.approx(raw_val, abs=5.0)
