"""Integration tests for PDF generation and CSV/JSON report exports."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.services.pdf_report_service import build_pdf_document  # noqa: E402
from pypdf import PdfReader  # noqa: E402


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]


def _app(tmp_path: Path):
    app = create_app(TestConfig)
    app.config["FORECAST_HISTORY_ROOT"] = str(tmp_path / "history")
    app.config["MODEL_ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    return app


@pytest.fixture()
def mock_features_dir(tmp_path: Path) -> Path:
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    target = 100 + rng.normal(0, 5, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": target,
            "year": ts.year,
            "month": ts.month,
            "day": ts.day,
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "quarter": ts.quarter,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "lag_1": pd.Series(target).shift(1),
            "lag_24": pd.Series(target).shift(24),
            "lag_168": pd.Series(target).shift(168),
            "rolling_mean_24": pd.Series(target).shift(1).rolling(24).mean(),
            "rolling_mean_168": pd.Series(target).shift(1).rolling(168).mean(),
        }
    ).dropna().reset_index(drop=True)

    n_clean = len(df)
    t_end = int(n_clean * 0.7)
    v_end = int(n_clean * 0.85)

    df.iloc[:t_end].to_parquet(fdir / "train.parquet", index=False)
    df.iloc[t_end:v_end].to_parquet(fdir / "validation.parquet", index=False)
    df.iloc[v_end:].to_parquet(fdir / "test.parquet", index=False)
    return fdir


def test_pdf_and_exports_for_test_evaluation_and_future(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # 1. Execute a test evaluation forecast
        resp = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
            },
        )
        assert resp.status_code == 200
        run_id = resp.get_json()["data"]["history"]["run_id"]

        # 2. Get report detail
        detail = client.get(f"/api/v1/reports/{run_id}")
        assert detail.status_code == 200
        data = detail.get_json()["data"]
        assert data["run_id"] == run_id
        assert "dataset_info" in data
        assert "validation_benchmark" in data
        assert "forecast_data" in data
        assert "predictions" not in data  # Contract: legacy raw predictions key must not be present at root

        # 3. Download PDF report
        pdf_resp = client.get(f"/api/v1/reports/{run_id}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.content_type == "application/pdf"
        assert pdf_resp.data.startswith(b"%PDF-")
        assert len(pdf_resp.data) > 1000

        # 4. Export CSV
        csv_resp = client.get(f"/api/v1/reports/{run_id}/export/csv")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.content_type
        csv_lines = csv_resp.data.decode("utf-8").strip().splitlines()
        assert len(csv_lines) > 1
        assert "timestamp,predicted,actual,residual" in csv_lines[0]

        # 5. Export JSON
        json_resp = client.get(f"/api/v1/reports/{run_id}/export/json")
        assert json_resp.status_code == 200
        json_data = json.loads(json_resp.data.decode("utf-8"))
        assert json_data["run_id"] == run_id

        # 6. Execute a future forecast
        fut_resp = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
                "horizon": 12,
            },
        )
        assert fut_resp.status_code == 200
        fut_run_id = fut_resp.get_json()["data"]["history"]["run_id"]

        # 7. Future PDF download
        fut_pdf = client.get(f"/api/v1/reports/{fut_run_id}/pdf")
        assert fut_pdf.status_code == 200
        assert fut_pdf.data.startswith(b"%PDF-")

        # 8. Future CSV export (Must NOT have actuals column!)
        fut_csv = client.get(f"/api/v1/reports/{fut_run_id}/export/csv")
        assert fut_csv.status_code == 200
        fut_csv_lines = fut_csv.data.decode("utf-8").strip().splitlines()
        assert "timestamp,predicted" == fut_csv_lines[0]
        assert "actual" not in fut_csv_lines[0]

        # 9. Nonexistent run 404
        missing_pdf = client.get("/api/v1/reports/nonexistent_12345678/pdf")
        assert missing_pdf.status_code == 404

        db.session.remove()
        db.drop_all()


def _extract_all_pdf_text(pdf_bytes: bytes) -> tuple[int, str]:
    """Helper to extract text from all pages of a PDF buffer."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return len(reader.pages), full_text


def test_pdf_dataset_intelligence_section_metrics():
    """Verify Dataset Intelligence & Data Quality Audit metrics appear accurately in extracted PDF text."""
    report = {
        "run_id": "audit_di_test_run_001",
        "domain": "retail",
        "model": "lightgbm",
        "target": "weekly_sales",
        "forecast_type": "test_evaluation",
        "timestamp": "2026-10-22 10:00:00 UTC",
        "dataset_intelligence": {
            "dataset_name": "walmart_store_sales.parquet",
            "original_row_count": 421570,
            "original_column_count": 16,
            "cleaned_row_count": 420000,
            "training_observations": 294000,
            "validation_observations": 63000,
            "test_observations": 63000,
            "forecast_horizon": 52,
            "detected_time_column": "Date",
            "detected_target_column": "Weekly_Sales",
            "canonical_target": "weekly_sales",
            "unique_timestamps": 143,
            "duplicate_rows_removed": 1570,
            "duplicate_timestamps_count": 0,
            "missing_values_handled": 12,
            "excluded_features": ["Store", "Dept", "IsHoliday_Raw"],
            "excluded_reasons": {
                "Store": "Entity identifier code excluded to prevent memorization",
                "Dept": "Department group identifier excluded",
            },
            "final_forecasting_features_count": 14,
            "domain_fit_score": 0.94,
            "domain_fit_label": "EXCELLENT",
            "suitability_status": "SUITABLE",
            "suitability_label": "Fully Suitable",
            "explanation": "Dataset contains consistent weekly retail records suitable for time-series demand forecasting.",
            "suitability_warnings": ["Minor missing values in promotional markdown fields"],
        },
        "forecast_data": {
            "points": [
                {"timestamp": "2025-01-01", "predicted": 25000.5, "actual": 24800.0, "residual": 200.5},
            ]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())
    assert num_pages >= 1

    # Verify Dataset Intelligence section title and key fields
    assert "Dataset Intelligence & Data Quality Audit" in text
    assert "walmart_store_sales.parquet" in text
    assert "421,570" in text
    assert "420,000" in text
    assert "294,000" in text
    assert "Date" in text
    assert "Weekly_Sales" in text
    assert "94.0%" in text
    assert "Fully Suitable" in text
    assert "1,570" in text
    assert "Store" in text
    assert "Dept" in text
    assert "prevent memorization" in text
    assert "Minor missing values in promotional markdown fields" in text


def test_pdf_100_points_forecast_full_multipage_table():
    """Verify a 100-point forecast includes all 100 observations across a multi-page table with repeat headers."""
    points = [
        {
            "timestamp": f"2025-06-01 {i:02d}:00:00",
            "predicted": 1000.0 + (i * 10.5),
            "lower_bound": 900.0 + (i * 10.5),
            "upper_bound": 1100.0 + (i * 10.5),
            "actual": 1005.0 + (i * 10.5),
            "residual": -5.0,
        }
        for i in range(100)
    ]

    report = {
        "run_id": "test_100_pts_run",
        "domain": "electricity",
        "model": "xgboost",
        "target": "demand_mw",
        "forecast_type": "test_evaluation",
        "horizon": 100,
        "forecast_data": {"points": points},
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())

    # 100 rows must span multiple pages (at least 3 pages)
    assert num_pages >= 3, f"Expected at least 3 pages for 100 points, got {num_pages}"

    # Verify observations from start, middle, and end appear in extracted text
    assert "2025-06-01 00:00:00" in text  # Point 0
    assert "2025-06-01 15:00:00" in text  # Point 15 (previously truncated!)
    assert "2025-06-01 50:00:00" in text  # Point 50
    assert "2025-06-01 99:00:00" in text  # Point 99 (last point!)

    # Verify no truncation disclaimer when all 100 points fit within threshold
    assert "Displaying all 100 forecast observations" in text
    assert "Displaying first 15 of 100 forecast points" not in text

    # Verify confidence bounds and actuals appear
    assert "Lower (95%)" in text
    assert "Upper (95%)" in text
    assert "Actual (Truth)" in text

    # Verify footer is present on pages
    assert "SmartXSense Academic Forecast Audit" in text


def test_pdf_xai_summary_shap_and_lime_explanations():
    """Verify XAI section with real SHAP attributions, LIME local explanation, and disclaimers."""
    report = {
        "run_id": "xai_shap_lime_run",
        "domain": "electricity",
        "model": "random_forest",
        "target": "demand_mw",
        "forecast_type": "test_evaluation",
        "explainability": {
            "selected_model": "Random Forest",
            "selection_reason": "Achieved lowest Validation RMSE (3.84 demand_mw) among evaluated candidates",
            "selection_criterion": "Lowest Validation RMSE",
            "validation_rmse": 3.84,
            "shap_features": [
                {"feature": "lag_1", "importance": 0.45},
                {"feature": "lag_24", "importance": 0.30},
                {"feature": "rolling_mean_24", "importance": 0.15},
                {"feature": "temperature_c", "importance": 0.10},
            ],
            "lime_explanation": {
                "sample_index": 12,
                "timestamp": "2025-01-15 14:00:00",
                "features": [
                    {"feature": "lag_1 > 120.5", "importance": 14.8},
                    {"feature": "temperature_c <= 18.0", "importance": -5.2},
                ],
            },
            "ai_explanation": {
                "explanation": "Predictions for electricity demand are strongly governed by diurnal recurrence and immediate lag persistence."
            },
        },
        "forecast_data": {
            "points": [{"timestamp": "2025-01-15 14:00:00", "predicted": 150.2, "actual": 148.5, "residual": 1.7}]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())
    assert num_pages >= 1

    # Check XAI section header and model selection
    assert "5. Explainable AI (XAI) Summary" in text
    assert "Random Forest" in text
    assert "Lowest Validation RMSE" in text
    assert "3.84 demand_mw" in text

    # Check SHAP global explanation
    assert "Tree SHAP" in text
    assert "lag_1" in text
    assert "lag_24" in text
    assert "Short-term temporal anchor" in text
    assert "Diurnal cycle regulator" in text

    # Check LIME local explanation
    assert "Local Prediction Explanation (LIME" in text
    assert "lag_1 > 120.5" in text
    assert "+14.8000" in text or "14.8" in text
    assert "Positive contribution" in text
    assert "Sample index #12" in text

    # Check Plain-English interpretation and disclaimers
    assert "Predictions for electricity demand are strongly governed" in text
    assert "Statistical Association vs. Causal Impact" in text
    assert "Global explanations reflect aggregate model behavior" in text


def test_pdf_xai_native_feature_importance_fallback():
    """Verify fallback to Native Model Feature Importance when SHAP is unavailable."""
    report = {
        "run_id": "native_imp_run",
        "domain": "retail",
        "model": "random_forest",
        "target": "sales",
        "forecast_type": "test_evaluation",
        "explainability": {
            "selected_model": "Random Forest",
            "validation_rmse": 12.5,
            "feature_importance": {
                "lag_1": 0.60,
                "lag_7": 0.25,
                "promo_flag": 0.15,
            },
        },
        "forecast_data": {
            "points": [{"timestamp": "2025-01-01", "predicted": 120.0, "actual": 118.0, "residual": 2.0}]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())
    assert "Native Model Feature Importance" in text
    assert "lag_1" in text
    assert "lag_7" in text
    assert "promo_flag" in text


def test_pdf_xai_unavailable_for_statistical_models():
    """Verify statistical models like ARIMA state explanation availability without crashing or fabricating."""
    report = {
        "run_id": "arima_run",
        "domain": "traffic",
        "model": "sarima",
        "target": "vehicle_flow",
        "forecast_type": "future",
        "horizon": 24,
        "forecast_data": {
            "points": [{"timestamp": f"2025-01-01 {i:02d}:00:00", "predicted": 500.0} for i in range(24)]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())
    assert "5. Explainable AI (XAI) Summary" in text
    assert "Sarima" in text
    assert "Not generated or unavailable for model architecture" in text
    assert "autoregressive" in text
    assert "lag coefficients" in text


def test_pdf_unknown_metrics_display_not_available_never_zero():
    """Verify that unknown/missing metrics render 'Not available' or 'N/A', never false zeros."""
    report = {
        "run_id": "minimal_empty_run",
        "domain": "retail",
        "model": "prophet",
        "target": "revenue",
        "forecast_type": "future",
        "dataset_intelligence": {},  # completely empty
        "forecast_data": {
            "points": [{"timestamp": "2025-01-01", "predicted": 100.0}]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())
    # Should say Not available for missing metrics
    assert "Not available" in text
    assert "Original Dimensions:" in text
    assert "Domain-Fit Score:" in text


def test_pdf_xai_clean_formatting_and_no_raw_markdown_or_internal_ids():
    """Verify clean XAI hierarchy, removal of raw markdown tokens/malformed newlines, and scrubbing of LLM IDs."""
    malformed_narrative = (
        "### n AI Forecast Summary\n"
        "Retail weekly sales indicate a stable seasonal trajectory. Model: openai/gpt-oss-120b executed.\n\n"
        "### nn Situation\n"
        "Demand is projected to remain steady without unexpected volatility.\n\n"
        "### n Why / Possible Causes\n"
        "The primary predictive signals include `lag_7` (**weekly lag**), `month` (*seasonal contribution*), "
        "`inflation_rate` (purchasing power indicator), `ewma_3` (recent smoothed trend), and `day_of_week`.\n\n"
        "### n Recommended Actions / Remedies\n"
        "- Optimize product replenishment and safety stock buffers.\n"
        "- Review promotional discount elasticity before month-end.\n\n"
        "### n Precautions\n"
        "- Continuously track incoming actual values against predicted intervals.\n"
        "- Trigger formal model reassessment and retraining if errors exceed tolerance.\n"
    )

    report = {
        "run_id": "clean_xai_test_001",
        "domain": "retail",
        "model": "xgboost",
        "target": "weekly_sales",
        "forecast_type": "test_evaluation",
        "metrics": {"rmse": 120.5, "mae": 95.2, "mape": 0.045, "r2": 0.91},
        "explainability": {
            "selected_model": "XGBoost",
            "selection_criterion": "Lowest Validation RMSE",
            "selection_reason": "Achieved lowest Validation RMSE (118.2 weekly_sales) across all evaluated candidates",
            "validation_rmse": 118.2,
            "ai_explanation": {
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "explanation": malformed_narrative,
            },
            "shap_features": [
                {"feature": "lag_7", "importance": 0.35},
                {"feature": "month", "importance": 0.22},
                {"feature": "inflation_rate", "importance": 0.18},
                {"feature": "ewma_3", "importance": 0.14},
                {"feature": "day_of_week", "importance": 0.11},
            ],
        },
        "forecast_data": {
            "points": [{"timestamp": "2025-01-01", "predicted": 2500.0, "actual": 2480.0, "residual": 20.0}]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())

    # 1. Verify clean section headings exist
    assert "AI Forecast Summary" in text
    assert "Situation Analysis" in text
    assert "Key Influencing Factors" in text
    assert "Recommended Actions" in text
    assert "Precautions and Monitoring" in text
    assert "Explainability Disclosures & Limitations" in text

    # 2. Verify malformed markdown headers and tokens are completely absent
    assert "###" not in text
    assert "### n" not in text
    assert "### nn" not in text
    assert "**" not in text

    # 3. Verify internal LLM provider and model identifiers do not appear in user-facing output
    assert "openai/gpt-oss-120b" not in text
    assert "gpt-oss-120b" not in text
    assert "openai/" not in text

    # 4. Verify actual forecasting model name is preserved
    assert "XGBoost" in text or "Xgboost" in text

    # 5. Verify feature findings are preserved
    assert "lag_7" in text
    assert "month" in text
    assert "inflation_rate" in text
    assert "ewma_3" in text
    assert "day_of_week" in text

    # 6. Verify disclosures are present and distinct
    assert "Statistical Association vs. Causal Impact" in text
    assert "Scope Distinction" in text
    assert "Directional Attribution vs. Importance Magnitude" in text


def test_pdf_future_forecast_clearly_states_unavailable_quantitative_metrics():
    """Verify future scenario forecast clearly notes that quantitative error metrics are not available."""
    report = {
        "run_id": "future_notice_run",
        "domain": "electricity",
        "model": "random_forest",
        "target": "demand_mw",
        "forecast_type": "future",
        "horizon": 24,
        "forecast_data": {
            "points": [{"timestamp": f"2025-01-01 {i:02d}:00:00", "predicted": 400.0} for i in range(24)]
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())

    # Verify notice that quantitative held-out metrics do not exist for future timestamps
    assert "Quantitative held-out error metrics" in text
    assert "future scenario horizons" in text or "future timestamps" in text


def test_frontend_dashboard_removes_internal_llm_model_label():
    """Verify that frontend ForecastDashboard.jsx renders clean AI Explanation heading without model labels or Automated Synthesis badge."""
    dashboard_file = ROOT / "frontend" / "src" / "components" / "forecasting" / "ForecastDashboard.jsx"
    content = dashboard_file.read_text(encoding="utf-8")

    # Assert internal model label was removed
    assert "forecastData.llm_explanation.model" not in content
    # Assert Automated Synthesis badge was completely removed
    assert "Automated Synthesis" not in content
    # Assert clean AI Explanation heading with icon is preserved
    assert "AI Explanation" in content
    assert "🤖" in content


def test_pdf_unicode_normalization_eliminates_black_squares():
    """Verify that PDF generation normalizes non-breaking hyphens (U+2011), curly quotes, and other
    unsupported Unicode glyphs into standard Type 1 equivalents, eliminating black squares (■ / U+25A0)."""
    report = {
        "run_id": "TEST_UNICODE_NORMALIZATION_RUN",
        "created_at": "2026-10-22T08:00:00Z",
        "domain": "retail",
        "forecast_type": "test_evaluation",
        "model": "random_forest",
        "target": "weekly_sales",
        "metrics": {"rmse": 142.1, "mae": 101.4, "mape": 0.045, "r2": 0.938},
        "points": [
            {"timestamp": "2026-07-01", "predicted": 1250.0, "actual": 1240.0, "residual": 10.0},
        ],
        "explainability": {
            "selected_model": "random_forest",
            "selection_criterion": "Lowest Validation RMSE",
            "selection_reason": "Achieved optimal bias-variance tradeoff across chronological cross-validation.",
            "validation_rmse": 142.1,
            "feature_importance": {
                "Higher\u2011tier cities": 0.42,
                "lower\u2011tier locations": 0.31,
                "carry\u2011over effect": 0.18,
                "promo_discount": 0.09,
            },
            "ai_explanation": {
                "explanation": (
                    "### AI Forecast Summary\n"
                    "Demand in Higher\u2011tier cities remains structurally resilient with strong baseline volume.\n\n"
                    "### Situation\n"
                    "In lower\u2011tier locations, we observe an enduring carry\u2011over effect from seasonal promotions \u2014 "
                    "a \u201cmeasured\u201d uplift of +12.4% with minimal volatility\u2026\n\n"
                    "### Recommended Actions\n"
                    "* Prioritize replenishment buffer stock in Higher\u2011tier cities.\n"
                    "* Calibrate margin elasticity for lower\u2011tier locations.\n\n"
                    "### Precautions and Monitoring\n"
                    "* Monitor carry\u2011over effect degradation if macro indicators soften."
                ),
            },
        },
    }

    buf = build_pdf_document(report)
    num_pages, text = _extract_all_pdf_text(buf.getvalue())

    # Assert completely zero black square artifacts in extracted PDF text
    assert "\u25a0" not in text, "Found black square U+25A0 glyph in extracted PDF text!"
    assert "■" not in text, "Found black square ■ glyph in extracted PDF text!"

    # Assert phrases render with standard safe hyphens
    assert "Higher-tier cities" in text, "Expected 'Higher-tier cities' with standard hyphen"
    assert "lower-tier locations" in text, "Expected 'lower-tier locations' with standard hyphen"
    assert "carry-over effect" in text, "Expected 'carry-over effect' with standard hyphen"
