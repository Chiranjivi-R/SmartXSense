"""Unit tests for SmartXSense LLM Explanation Service.

Tests:
- Groq primary provider success & fallback on missing key / API error.
- Ollama fallback success & model availability check warning.
- Deterministic fallback when both Groq and Ollama fail/are unavailable.
- Strict compliance with R² interpretation, MAPE scale context, duplicate timestamps,
  and forbidden PII feature exclusion.
- Dataset identity binding and isolation across datasets.
- Non-blocking LLM failure during forecasting execution.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest

from backend.app.services.llm_service import (
    FORBIDDEN_DRIVER_KEYWORDS,
    check_ollama_model_available,
    generate_deterministic_explanation,
    generate_explanation,
)
from backend.app.services.ml_service import forecast_service


from backend.app import create_app


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]



@pytest.fixture
def app():
    test_app = create_app(TestConfig)
    with test_app.app_context():
        yield test_app


@pytest.fixture
def mock_context():

    return {
        "dataset_id": "dataset_alpha",
        "dataset_hash": "hash_alpha_123",
        "dataset_name": "Retail Transactions Alpha",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "model_selection_reason": "won validation RMSE comparison",
        "metrics": {
            "rmse": 12.45,
            "mae": 8.30,
            "mape": 0.15,
            "r2": 0.84,
            "smape": 0.14,
            "wape": 0.13,
        },
        "baseline_metrics": {
            "naive_rmse": 25.10,
            "model_improvement_pct": 50.4,
        },
        "data_quality": {
            "duplicate_rows_removed": 5,
            "duplicate_timestamps_handled": True,
            "missing_values_handled": 12,
            "excluded_features": ["customer_id", "email", "phone", "customer_name"],
        },
        "final_model_features": ["rolling_mean_7", "lag_1", "cpi_clothing"],
        "shap_features": [
            {"feature": "rolling_mean_7", "importance": 0.45},
            {"feature": "lag_1", "importance": 0.30},
        ],
        "target_scale_warning": False,
    }


def test_groq_primary_success(mock_context):
    """Test Groq primary provider when API key is provided and API responds successfully."""
    with patch("backend.app.services.llm_service._get_config_var") as mock_cfg, \
         patch("backend.app.services.llm_service.call_groq_api") as mock_groq:
        
        def cfg_side_effect(key, default=None):
            if key == "GROQ_API_KEY":
                return "gsk_test_mock_key_123"
            if key == "GROQ_MODEL":
                return "llama-3.3-70b-versatile"
            return default
            
        mock_cfg.side_effect = cfg_side_effect
        mock_groq.return_value = "Groq narrative explanation for Dataset Alpha."
        
        result = generate_explanation(mock_context)
        
        assert result["provider"] == "groq"
        assert result["model"] == "llama-3.3-70b-versatile"
        assert result["fallback_used"] is False
        assert "Groq narrative" in result["explanation"]
        assert result["dataset_id"] == "dataset_alpha"
        assert result["dataset_hash"] == "hash_alpha_123"


def test_groq_missing_key_fallback_to_ollama(mock_context):
    """Test fallback to Ollama when GROQ_API_KEY is missing/empty."""
    with patch("backend.app.services.llm_service._get_config_var") as mock_cfg, \
         patch("backend.app.services.llm_service.check_ollama_model_available", return_value=True), \
         patch("backend.app.services.llm_service.call_ollama_api", return_value="Ollama local narrative."):
        
        def cfg_side_effect(key, default=None):
            if key == "GROQ_API_KEY":
                return None
            if key == "OLLAMA_BASE_URL":
                return "http://localhost:11434"
            if key == "OLLAMA_MODEL":
                return "qwen3:8b"
            return default
            
        mock_cfg.side_effect = cfg_side_effect
        
        result = generate_explanation(mock_context)
        
        assert result["provider"] == "ollama"
        assert result["model"] == "qwen3:8b"
        assert result["fallback_used"] is True
        assert "Ollama local narrative" in result["explanation"]


def test_ollama_unavailable_warning_and_deterministic_fallback(mock_context, caplog):
    """Test backend warning log when Ollama qwen3:8b is unavailable."""
    with patch("backend.app.services.llm_service._get_config_var") as mock_cfg, \
         patch("requests.get") as mock_get:
        
        def cfg_side_effect(key, default=None):
            if key == "GROQ_API_KEY":
                return None
            if key == "OLLAMA_BASE_URL":
                return "http://localhost:11434"
            if key == "OLLAMA_MODEL":
                return "qwen3:8b"
            return default
            
        mock_cfg.side_effect = cfg_side_effect
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama2:latest"}]}
        
        with caplog.at_level(logging.WARNING):
            avail = check_ollama_model_available("http://localhost:11434", "qwen3:8b")
            assert avail is False
            assert "Ollama fallback model qwen3:8b is not available." in caplog.text
            
            result = generate_explanation(mock_context)
            assert result["provider"] == "deterministic"
            assert result["model"] is None
            assert result["fallback_used"] is True
            assert result["status"] == "fallback"


def test_both_providers_unavailable_returns_deterministic(mock_context):
    """Test deterministic fallback when both Groq API errors and Ollama is unreachable."""
    with patch("backend.app.services.llm_service._get_config_var") as mock_cfg, \
         patch("backend.app.services.llm_service.call_groq_api", return_value=None), \
         patch("backend.app.services.llm_service.check_ollama_model_available", return_value=False):
        
        mock_cfg.return_value = None
        result = generate_explanation(mock_context)
        
        assert result["provider"] == "deterministic"
        assert result["model"] is None
        assert result["fallback_used"] is True
        assert ("Random Forest" in result["explanation"] or "AI Forecast Summary" in result["explanation"])


def test_negative_r2_explanation_integrity():
    """Verify that negative R² is NEVER called accuracy and explains poor test generalization."""
    ctx = {
        "dataset_id": "ds_neg_r2",
        "domain": "retail",
        "target": "sales",
        "selected_model": "XGBoost",
        "metrics": {"r2": -0.019, "rmse": 100.0, "mae": 80.0},
        "baseline_metrics": {"naive_rmse": 90.0, "model_improvement_pct": -11.0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "accuracy" not in explanation.lower()
    assert "negative r²" in explanation.lower()
    assert "quality needs improvement" in explanation.lower() or "did not explain the variation" in explanation.lower()


def test_positive_r2_explanation():
    """Verify positive R² explains variance percentage without calling it accuracy."""
    ctx = {
        "dataset_id": "ds_pos_r2",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "metrics": {"r2": 0.84, "rmse": 12.0, "mae": 9.0},
        "baseline_metrics": {"naive_rmse": 30.0, "model_improvement_pct": 60.0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "accuracy" not in explanation.lower()
    assert "explains approximately 84.0% of the variation" in explanation or "84" in explanation


def test_retail_increasing_demand_situation():
    """Verify retail increasing demand recommends inventory replenishment."""
    ctx = {
        "dataset_id": "ds_retail_inc",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "forecast_trend": "increasing",
        "metrics": {"r2": 0.75, "rmse": 10.0, "mae": 7.0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "### 🧠 AI Forecast Summary" in explanation
    assert "### ⚠️ Situation" in explanation
    assert "### 💡 Recommended Actions / Remedies" in explanation
    assert "Increase inventory" in explanation or "replenishment" in explanation


def test_retail_declining_demand_situation():
    """Verify retail declining demand advises adjusting procurement down."""
    ctx = {
        "dataset_id": "ds_retail_dec",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "forecast_trend": "decreasing",
        "metrics": {"r2": 0.75, "rmse": 10.0, "mae": 7.0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "Adjust procurement down" in explanation or "excessive inventory" in explanation


def test_electricity_peak_demand_situation():
    """Verify electricity peak demand recommends generation capacity preparation."""
    ctx = {
        "dataset_id": "ds_elec_peak",
        "domain": "electricity",
        "target": "demand_mw",
        "selected_model": "Random Forest",
        "forecast_trend": "highly_variable",
        "metrics": {"r2": 0.80, "rmse": 50.0, "mae": 35.0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "generation capacity" in explanation or "reserve capacity" in explanation


def test_no_false_duplicate_timestamp_statement_when_duplicates_zero():
    """Verify duplicate timestamp statement is excluded when duplicates handled is False."""
    ctx = {
        "dataset_id": "clean_ds",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "metrics": {"r2": 0.85, "rmse": 10.0, "mae": 8.0},
        "data_quality": {"duplicate_timestamps_handled": False, "duplicate_rows_removed": 0},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "Multiple transaction records with identical timestamps were automatically aggregated" not in explanation


def test_duplicate_timestamps_explained_when_present():
    """Verify duplicate timestamps are described when present in metadata."""
    ctx = {
        "dataset_id": "ds_dup_ts",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "metrics": {"rmse": 10.0},
        "data_quality": {"duplicate_timestamps_handled": True, "duplicate_timestamps_count": 15},
    }
    explanation = generate_deterministic_explanation(ctx)

    assert "Multiple transaction records with identical timestamps were automatically aggregated" in explanation


def test_forbidden_pii_features_never_in_explanation():
    """Verify PII fields (Customer ID, Name, Email, DOB) are strictly filtered out of drivers."""
    ctx = {
        "dataset_id": "ds_pii",
        "domain": "retail",
        "target": "sales",
        "selected_model": "Random Forest",
        "metrics": {"rmse": 10.0},
        "data_quality": {
            "excluded_features": ["customer_id", "email", "phone", "first_name", "date_of_birth"]
        },
        "shap_features": [
            {"feature": "customer_id"},
            {"feature": "email"},
            {"feature": "rolling_mean_7"},
        ],
    }
    explanation = generate_deterministic_explanation(ctx)

    for forbidden in ["customer_id", "email", "phone", "first_name", "date_of_birth"]:
        assert forbidden not in explanation


def test_dataset_identity_isolation():
    """Verify explanations for Dataset A -> B -> A maintain dataset identity and isolation."""
    ctx_a = {"dataset_id": "Dataset_A", "dataset_hash": "hash_AAAA", "domain": "retail", "selected_model": "Random Forest", "metrics": {"rmse": 10}}
    ctx_b = {"dataset_id": "Dataset_B", "dataset_hash": "hash_BBBB", "domain": "electricity", "selected_model": "XGBoost", "metrics": {"rmse": 50}}

    with patch("backend.app.services.llm_service.call_groq_api", return_value="Mock narrative"):
        res_a1 = generate_explanation(ctx_a)
        res_b = generate_explanation(ctx_b)
        res_a2 = generate_explanation(ctx_a)

        assert res_a1["dataset_id"] == "Dataset_A"
        assert res_a1["dataset_hash"] == "hash_AAAA"
        assert res_b["dataset_id"] == "Dataset_B"
        assert res_b["dataset_hash"] == "hash_BBBB"
        assert res_a2["dataset_id"] == "Dataset_A"
        assert res_a2["dataset_hash"] == "hash_AAAA"


def test_llm_failure_does_not_break_forecasting(app):
    """Verify that forecasting attaches fallback explanation even if LLM service raises an exception."""
    with patch("backend.app.services.llm_service.generate_explanation", side_effect=RuntimeError("LLM crashed!")):
        res = {
            "model": "random_forest",
            "metrics": {"rmse": 10.5, "mae": 7.2},
            "baseline_metrics": {"rmse": 20.0},
        }
        payload = {"domain": "retail", "model": "random_forest"}
        dataset_ctx = {"dataset_id": "demo_retail", "dataset_hash": "demo_hash"}

        from backend.app.services.ml_service import _attach_llm_explanation
        out = _attach_llm_explanation(res, payload, dataset_ctx)

        assert "llm_explanation" in out
        assert out["llm_explanation"]["provider"] == "deterministic"
        assert out["llm_explanation"]["status"] == "fallback"


