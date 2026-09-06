"""Integration tests for backend forecasting, model comparison, and explainability endpoints."""

from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]


@pytest.fixture()
def client(tmp_path):
    """Flask test client fixture — yields test_client with isolated ML_ARTIFACTS_ROOT."""
    class CustomTestConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")


    app = create_app(CustomTestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def mock_features_dir(tmp_path):
    """Create minimal feature split artifacts in tmp_path for test use."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": np.sin(np.linspace(0, 10, n)) * 100 + 500,
        "sales": np.sin(np.linspace(0, 10, n)) * 10 + 50,
        "feature_lag1": np.roll(np.sin(np.linspace(0, 10, n)), 1),
    })

    df.iloc[:35].to_parquet(features_dir / "train.parquet", index=False)
    df.iloc[35:48].to_parquet(features_dir / "validation.parquet", index=False)
    df.iloc[48:].to_parquet(features_dir / "test.parquet", index=False)

    return features_dir


# ---------------------------------------------------------------------------
# Forecasting endpoint tests
# ---------------------------------------------------------------------------

def test_forecast_endpoint_single_model(client, mock_features_dir):
    """POST /train returns 200 with validation and test metrics for RF."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["model"] == "random_forest"
    assert "validation_metrics" in json_data["data"]
    assert "test_metrics" in json_data["data"]
    assert "predictions" in json_data["data"]
    assert isinstance(json_data["data"]["predictions"], list)
    assert len(json_data["data"]["predictions"]) > 0


def test_forecast_endpoint_hybrid(client, mock_features_dir):
    """POST /train with hybrid returns 200 with weights and ensemble metrics."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "hybrid",
            "components": ["arima", "random_forest"],
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["model"] == "hybrid"
    assert "weights" in json_data["data"]
    assert "validation_metrics" in json_data["data"]


def test_compare_endpoint(client, mock_features_dir):
    """POST /compare returns ranked models using validation metrics only."""
    # Use a lightweight explicit candidate list for fast test execution
    response = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "models": ["random_forest", "xgboost"],
            "selection_metric": "rmse",
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["selection_metric"] == "rmse"
    assert data["selection_split"] == "validation"
    assert "recommended_model" in data
    assert isinstance(data["ranking"], list)
    assert len(data["ranking"]) >= 2


def test_compare_selection_split_is_validation(client, mock_features_dir):
    """Confirm that comparison output always declares selection_split = 'validation'."""
    response = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "models": ["random_forest", "xgboost"],
            "selection_metric": "rmse",
        },
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["selection_split"] == "validation"
    # Recommended model must appear in ranking
    recommended = data["recommended_model"]
    ranked_models = [r["model"] for r in data["ranking"]]
    assert recommended in ranked_models


def test_compare_benchmark_cache_and_reuse(client, mock_features_dir):
    """Confirm first compare computes benchmark and second identical call reuses cached benchmark."""
    payload = {
        "domain": "electricity",
        "features_dir": str(mock_features_dir),
        "models": ["random_forest", "xgboost"],
        "selection_metric": "mae",
    }
    # First benchmark call
    res1 = client.post("/api/v1/forecasting/compare", json=payload)
    assert res1.status_code == 200
    data1 = res1.get_json()["data"]
    assert "benchmark_execution" in data1
    assert data1["benchmark_execution"]["reused"] is False
    assert data1["benchmark_execution"]["status"] == "Benchmark Computed"

    # Second benchmark call -> reuses benchmark cache
    res2 = client.post("/api/v1/forecasting/compare", json=payload)
    assert res2.status_code == 200
    data2 = res2.get_json()["data"]
    assert "benchmark_execution" in data2
    assert data2["benchmark_execution"]["reused"] is True
    assert data2["benchmark_execution"]["status"] == "Benchmark Reused"
    assert data2["recommended_model"] == data1["recommended_model"]


def test_compare_dataset_fingerprint_change_invalidates_cache(client, mock_features_dir):
    """Confirm that modifying dataset features invalidates benchmark cache reuse."""
    payload = {
        "domain": "electricity",
        "features_dir": str(mock_features_dir),
        "models": ["random_forest", "xgboost"],
        "selection_metric": "rmse",
    }
    res1 = client.post("/api/v1/forecasting/compare", json=payload)
    assert res1.status_code == 200
    data1 = res1.get_json()["data"]
    assert data1["benchmark_execution"]["reused"] is False

    # Modify train.parquet
    train_path = mock_features_dir / "train.parquet"
    df = pd.read_parquet(train_path)
    df["demand_mw"] = df["demand_mw"] * 1.5
    df.to_parquet(train_path, index=False)

    res2 = client.post("/api/v1/forecasting/compare", json=payload)
    assert res2.status_code == 200
    data2 = res2.get_json()["data"]
    assert data2["benchmark_execution"]["reused"] is False
    assert data2["benchmark_execution"]["status"] == "Benchmark Computed"


# ---------------------------------------------------------------------------
# Path security tests
# ---------------------------------------------------------------------------

def test_path_security_traversal_rejection(client):
    """Paths with '..' must be rejected with 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": "../../../secret_dir",
            "model": "random_forest",
        },
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "traversal" in json_data["message"].lower()


def test_path_security_nonexistent_dir(client):
    """Paths pointing to non-existent directories must return 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": "C:/nonexistent_test_path_abc123",
            "model": "random_forest",
        },
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"


# ---------------------------------------------------------------------------
# Invalid input tests
# ---------------------------------------------------------------------------

def test_invalid_domain_or_model(client, mock_features_dir):
    """Invalid domain or unsupported model/domain combo must return 400."""
    res_domain = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "invalid_domain",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
        },
    )
    assert res_domain.status_code == 400

    # 1. Unsuitable model structure (trend_xgboost without hourly lags) must return 400
    res_inapplicable = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "retail",
            "features_dir": str(mock_features_dir),
            "model": "trend_xgboost",
        },
    )
    assert res_inapplicable.status_code == 400

    # 2. Suitable single-series retail ARIMA is now accepted (200)
    res_arima = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "retail",
            "features_dir": str(mock_features_dir),
            "model": "arima",
        },
    )
    assert res_arima.status_code == 200


def test_invalid_json_payload(client):
    """Missing or malformed JSON body must return 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        data="not-json",
        content_type="text/plain",
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Explainability endpoint tests
# ---------------------------------------------------------------------------

def test_explain_endpoint_global_shap(client, mock_features_dir):
    """POST /explain with global type returns feature importances dict."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "explanation_type": "global",
            "method": "shap",
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["explanation_type"] == "global"
    assert data["method"] == "shap"
    assert isinstance(data["feature_importance"], dict)
    assert len(data["feature_importance"]) > 0


def test_explain_endpoint_local_shap(client, mock_features_dir):
    """POST /explain with local type returns per-sample SHAP attribution."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "xgboost",
            "explanation_type": "local",
            "method": "shap",
            "sample_index": 0,
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["explanation_type"] == "local"
    assert isinstance(data["prediction"], float)
    assert "shap_contributions" in data
    assert "top_positive_contributors" in data
    assert "top_negative_contributors" in data


def test_explain_endpoint_unsupported_model(client, mock_features_dir):
    """Unsupported explainability models must return 400."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "arima",
            "explanation_type": "global",
        },
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    # Error message must indicate explainability limitation
    assert "explainability" in json_data["message"].lower() or "supported" in json_data["message"].lower()


def test_explain_endpoint_lstm_supported(client, mock_features_dir):
    """LSTM model explainability request succeeds with 200 OK."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "lstm",
            "explanation_type": "global",
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Prediction structure & prediction_limit tests
# ---------------------------------------------------------------------------

def test_forecast_endpoint_returns_chart_predictions(client, mock_features_dir):
    """Forecast response must return predictions array with timestamp, actual, predicted."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 10,
            "prediction_limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert "predictions" in data
    assert "prediction_count" in data
    assert "total_test_predictions" in data

    preds = data["predictions"]
    assert len(preds) == 5
    assert data["prediction_count"] == 5
    assert data["total_test_predictions"] == 12  # mock_features_dir test split has 12 rows (60-48)

    for item in preds:
        assert "timestamp" in item
        assert "actual" in item
        assert "predicted" in item
        assert isinstance(item["timestamp"], str)
        assert isinstance(item["actual"], (int, float))
        assert isinstance(item["predicted"], (int, float))
        assert np.isfinite(item["actual"])
        assert np.isfinite(item["predicted"])

    # Verify chronological ordering
    timestamps = [p["timestamp"] for p in preds]
    assert timestamps == sorted(timestamps)


def test_forecast_endpoint_prediction_limit_validation(client, mock_features_dir):
    """Invalid prediction_limit values (0, negative, >1000, bool, invalid type) return 400."""
    invalid_limits = [0, -5, 1500, True, "invalid_string"]
    for val in invalid_limits:
        res = client.post(
            "/api/v1/forecasting/train",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "prediction_limit": val,
            },
        )
        assert res.status_code == 400, f"Expected 400 for prediction_limit={val}"


def test_forecast_endpoint_trains_then_loads_artifact(tmp_path, mock_features_dir):
    """First forecast trains/saves; identical second forecast loads the artifact."""
    class ArtifactTestConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "ml_artifacts")

    app = create_app(ArtifactTestConfig)
    with app.app_context():
        db.create_all()
        test_client = app.test_client()

        payload = {
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 15,
            "prediction_limit": 20,
        }

        first = test_client.post("/api/v1/forecasting/forecast", json=payload)
        assert first.status_code == 200
        first_data = first.get_json()["data"]
        assert first_data["artifact"]["source"] == "trained"
        assert first_data["artifact"]["used"] is False
        assert "test_metrics" in first_data
        assert len(first_data["predictions"]) > 0

        second = test_client.post("/api/v1/forecasting/forecast", json=payload)
        assert second.status_code == 200
        second_data = second.get_json()["data"]
        assert second_data["artifact"]["source"] == "loaded"
        assert second_data["artifact"]["used"] is True
        assert second_data["test_metrics"]["rmse"] == pytest.approx(
            first_data["test_metrics"]["rmse"], rel=1e-9, abs=1e-9
        )
        assert second_data["prediction_count"] == first_data["prediction_count"]

        db.session.remove()
        db.drop_all()


def test_forecast_endpoint_hybrid_does_not_claim_artifact_load(tmp_path, mock_features_dir):
    """Hybrid always trains; persistence reuse is intentionally unsupported."""
    class ArtifactTestConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "ml_artifacts_hybrid")

    app = create_app(ArtifactTestConfig)
    with app.app_context():
        db.create_all()
        test_client = app.test_client()
        payload = {
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "hybrid",
            "components": ["random_forest", "xgboost"],
            "n_estimators": 10,
        }
        first = test_client.post("/api/v1/forecasting/forecast", json=payload)
        assert first.status_code == 200
        assert first.get_json()["data"]["artifact"]["source"] == "trained"

        second = test_client.post("/api/v1/forecasting/forecast", json=payload)
        assert second.status_code == 200
        assert second.get_json()["data"]["artifact"]["source"] == "trained"

        db.session.remove()
        db.drop_all()


def test_future_endpoint_valid_horizon(tmp_path, mock_features_dir):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "future_artifacts")

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        res = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 15,
                "horizon": 24,
            },
        )
        assert res.status_code == 200
        data = res.get_json()["data"]
        assert data["forecast_type"] == "future"
        assert data["horizon"] == 24
        assert len(data["predictions"]) == 24
        assert data["prediction_count"] == 24
        for item in data["predictions"]:
            assert set(item.keys()) == {"timestamp", "predicted"}
            assert "actual" not in item
            assert np.isfinite(item["predicted"])
        timestamps = [pd.Timestamp(p["timestamp"]) for p in data["predictions"]]
        assert timestamps == sorted(timestamps)
        hist_test = pd.read_parquet(Path(mock_features_dir) / "test.parquet")
        last_hist = pd.Timestamp(pd.to_datetime(hist_test["timestamp"], utc=True).max())
        assert timestamps[0] > last_hist
        db.session.remove()
        db.drop_all()


@pytest.mark.parametrize("bad_horizon", [0, -1, True, 7.5, "12", 2161])
def test_future_endpoint_rejects_invalid_horizon(tmp_path, mock_features_dir, bad_horizon):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "future_bad_horizon")

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        res = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "horizon": bad_horizon,
            },
        )
        assert res.status_code == 400
        db.session.remove()
        db.drop_all()


def test_future_endpoint_unsupported_model(tmp_path, mock_features_dir):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "future_unsup")

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        res = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "lstm",
                "horizon": 24,
            },
        )
        assert res.status_code == 400
        assert "not currently supported" in res.get_json()["message"]
        db.session.remove()
        db.drop_all()


def test_future_endpoint_artifact_reuse(tmp_path, mock_features_dir):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "future_reuse")

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        payload = {
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 12,
            "horizon": 12,
        }
        first = client.post("/api/v1/forecasting/future", json=payload)
        assert first.status_code == 200
        assert first.get_json()["data"]["artifact"]["source"] == "trained"

        second = client.post("/api/v1/forecasting/future", json=payload)
        assert second.status_code == 200
        assert second.get_json()["data"]["artifact"]["source"] == "loaded"
        assert second.get_json()["data"]["artifact"]["used"] is True
        db.session.remove()
        db.drop_all()


# ---------------------------------------------------------------------------
# Forecast history (Phase D)
# ---------------------------------------------------------------------------

def _history_app(tmp_path):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "hist_artifacts")
        FORECAST_HISTORY_ROOT = str(tmp_path / "hist_store")

    return create_app(Cfg)


def test_forecast_creates_history_and_lists_newest_first(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        payload = {
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 10,
            "prediction_limit": 20,
        }
        first = client.post("/api/v1/forecasting/forecast", json=payload)
        assert first.status_code == 200
        first_data = first.get_json()["data"]
        assert first_data["artifact"]["source"] == "trained"
        assert first_data["history"]["recorded"] is True
        run1 = first_data["history"]["run_id"]

        second = client.post("/api/v1/forecasting/forecast", json=payload)
        assert second.status_code == 200
        second_data = second.get_json()["data"]
        assert second_data["artifact"]["source"] == "loaded"
        run2 = second_data["history"]["run_id"]

        hist = client.get("/api/v1/forecasting/history?limit=20")
        assert hist.status_code == 200
        runs = hist.get_json()["data"]["runs"]
        assert len(runs) >= 2
        assert runs[0]["run_id"] == run2
        assert runs[1]["run_id"] == run1
        assert runs[0]["forecast_type"] == "test_evaluation"
        assert runs[0]["artifact_source"] == "loaded"
        assert runs[1]["artifact_source"] == "trained"
        assert "predictions" not in runs[0]
        assert runs[0]["features_id"] == mock_features_dir.name

        one = client.get(f"/api/v1/forecasting/history/{run2}")
        assert one.status_code == 200
        detail = one.get_json()["data"]
        assert detail["run_id"] == run2
        assert "predictions" not in detail
        assert detail["model"] == "random_forest"
        assert "result_ref" in detail

        db.session.remove()
        db.drop_all()


def test_future_forecast_creates_history(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 10,
                "horizon": 12,
            },
        )
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["history"]["recorded"] is True
        run_id = data["history"]["run_id"]

        hist = client.get("/api/v1/forecasting/history?domain=electricity")
        assert hist.status_code == 200
        runs = hist.get_json()["data"]["runs"]
        assert any(r["run_id"] == run_id and r["forecast_type"] == "future" and r["horizon"] == 12 for r in runs)
        db.session.remove()
        db.drop_all()


def test_failed_forecast_does_not_create_history(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        before = client.get("/api/v1/forecasting/history")
        assert before.status_code == 200
        before_count = len(before.get_json()["data"]["runs"])

        bad = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "retail",
                "features_dir": str(mock_features_dir),
                "model": "trend_xgboost",
            },
        )
        assert bad.status_code == 400

        after = client.get("/api/v1/forecasting/history")
        assert after.status_code == 200
        assert len(after.get_json()["data"]["runs"]) == before_count
        db.session.remove()
        db.drop_all()


def test_history_filters_and_limit_validation(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        for model in ("random_forest", "xgboost"):
            res = client.post(
                "/api/v1/forecasting/forecast",
                json={
                    "domain": "electricity",
                    "features_dir": str(mock_features_dir),
                    "model": model,
                    "n_estimators": 8,
                    "prediction_limit": 10,
                },
            )
            assert res.status_code == 200

        by_model = client.get("/api/v1/forecasting/history?model=xgboost&limit=10")
        assert by_model.status_code == 200
        runs = by_model.get_json()["data"]["runs"]
        assert len(runs) >= 1
        assert all(r["model"] == "xgboost" for r in runs)

        by_domain = client.get("/api/v1/forecasting/history?domain=electricity&limit=10")
        assert by_domain.status_code == 200
        assert all(r["domain"] == "electricity" for r in by_domain.get_json()["data"]["runs"])

        for bad_limit in ("0", "-1", "true", "abc", "101"):
            bad = client.get(f"/api/v1/forecasting/history?limit={bad_limit}")
            assert bad.status_code == 400

        bad_domain = client.get("/api/v1/forecasting/history?domain=spaceships")
        assert bad_domain.status_code == 400
        db.session.remove()
        db.drop_all()


def test_history_invalid_run_id_404(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        missing = client.get("/api/v1/forecasting/history/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert missing.status_code == 404
        traversal = client.get("/api/v1/forecasting/history/../etc/passwd")
        assert traversal.status_code in (404, 400)
        db.session.remove()
        db.drop_all()


def test_history_auto_select_records_actual_model(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        # Frontend Auto Select resolves via /compare then forecasts the recommended model.
        compare = client.post(
            "/api/v1/forecasting/compare",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "models": ["random_forest", "xgboost"],
                "selection_metric": "rmse",
            },
        )
        assert compare.status_code == 200
        recommended = compare.get_json()["data"]["recommended_model"]
        assert recommended in {"random_forest", "xgboost"}

        forecast = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": recommended,
                "n_estimators": 8,
                "selection_method": "validation_rmse",
            },
        )
        assert forecast.status_code == 200
        data = forecast.get_json()["data"]
        assert data["model"] == recommended
        run_id = data["history"]["run_id"]

        detail = client.get(f"/api/v1/forecasting/history/{run_id}")
        assert detail.status_code == 200
        body = detail.get_json()["data"]
        assert body["model"] == recommended
        assert body["model"] != "auto_select"
        assert body["selection_method"] == "validation_rmse"
        db.session.remove()
        db.drop_all()


def test_history_delete_single_run(tmp_path, mock_features_dir):
    app = _history_app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        forecast = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
            },
        )
        assert forecast.status_code == 200
        run_id = forecast.get_json()["data"]["history"]["run_id"]

        deleted = client.delete(f"/api/v1/forecasting/history/{run_id}")
        assert deleted.status_code == 200
        assert deleted.get_json()["data"]["deleted"] is True

        missing = client.get(f"/api/v1/forecasting/history/{run_id}")
        assert missing.status_code == 404
        db.session.remove()
        db.drop_all()


def test_forecast_with_demo_dataset_key(client):
    response = client.post(
        "/api/v1/forecasting/forecast",
        json={
            "domain": "retail",
            "features_dir": "demo_retail",
            "model": "random_forest",
            "n_estimators": 5,
        },
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["model"] == "random_forest"
    assert data["domain"] == "retail"
    assert "dataset_context" in data
    assert data["dataset_context"]["dataset_name"] == "Retail Sales Demo"


def test_forecast_artifact_reuse_and_fingerprint_isolation(client):
    # First call trains model
    r1 = client.post(
        "/api/v1/forecasting/forecast",
        json={
            "domain": "retail",
            "features_dir": "demo_retail",
            "model": "random_forest",
            "n_estimators": 5,
        },
    )
    assert r1.status_code == 200

    # Second call with identical params reuses artifact
    r2 = client.post(
        "/api/v1/forecasting/forecast",
        json={
            "domain": "retail",
            "features_dir": "demo_retail",
            "model": "random_forest",
            "n_estimators": 5,
        },
    )
    assert r2.status_code == 200
    assert r2.get_json()["data"]["execution"]["artifact_reused"] is True


def test_lstm_and_hybrid_api_endpoints(client, mock_features_dir):
    """Verify forecast, compare, and explain endpoints for LSTM and Hybrid."""
    # LSTM forecast
    lstm_res = client.post(
        "/api/v1/forecasting/forecast",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "lstm",
            "epochs": 1,
            "batch_size": 16,
        },
    )
    assert lstm_res.status_code == 200
    lstm_data = lstm_res.get_json()["data"]
    assert lstm_data["model"] == "lstm"
    assert "dataset_context" in lstm_data

    # Hybrid forecast
    hybrid_res = client.post(
        "/api/v1/forecasting/forecast",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "hybrid",
            "components": ["random_forest", "xgboost", "lstm"],
            "n_estimators": 5,
            "epochs": 1,
        },
    )
    assert hybrid_res.status_code == 200
    hybrid_data = hybrid_res.get_json()["data"]
    assert hybrid_data["model"] == "hybrid"
    assert "weights" in hybrid_data

    # Compare endpoint with all candidate models
    compare_res = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "models": ["random_forest", "xgboost", "lstm", "hybrid"],
            "selection_metric": "rmse",
        },
    )
    assert compare_res.status_code == 200
    comp_data = compare_res.get_json()["data"]
    assert "ranking" in comp_data
    assert len(comp_data["ranking"]) >= 3
    assert comp_data["recommended_model"] in ["random_forest", "xgboost", "lstm", "hybrid"]

    # Explain endpoint for LSTM and Hybrid
    exp_lstm = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "lstm",
        },
    )
    assert exp_lstm.status_code == 200

    exp_hybrid = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "hybrid",
        },
    )
    assert exp_hybrid.status_code == 200


def test_dataset_identity_isolation_and_invalid_dataset_id(client):
    """Test that invalid dataset IDs return 404 and dataset context returns dataset_id and dataset_hash."""
    # Invalid dataset ID must return 404 error instead of silently falling back
    res_invalid = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "retail",
            "dataset_id": "non_existent_dataset_9999",
            "model": "random_forest",
        },
    )
    assert res_invalid.status_code == 404

    # Demo dataset returns proper dataset_context with dataset_id and dataset_hash
    res_demo = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "retail",
            "dataset_id": "demo_retail",
            "model": "random_forest",
            "n_estimators": 5,
        },
    )
    assert res_demo.status_code == 200
    ctx = res_demo.get_json()["data"]["dataset_context"]
    assert ctx["dataset_id"] == "demo_retail"
    assert "dataset_hash" in ctx
    assert "target" in ctx


def test_dataset_a_dataset_b_dataset_a_switching_isolation(client):
    """Verify complete runtime data isolation when switching Dataset A -> Dataset B -> Dataset A."""
    import io

    test_client, _ = client if isinstance(client, tuple) else (client, None)

    # Register and get auth token
    test_client.post("/api/v1/auth/register", json={"email": "iso_test@example.com", "password": "SecurePassword123!", "full_name": "Iso User", "role": "data_scientist"})
    login_res = test_client.post("/api/v1/auth/login", json={"email": "iso_test@example.com", "password": "SecurePassword123!"})
    token = login_res.get_json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Prepare Dataset A (sales ~1,000)
    csv_a = b"Date,Sales,Discount,City\n" + b"\n".join(
        f"2026-01-{(i%28)+1:02d},{1000 + i * 5},0.1,Mumbai".encode("utf-8") for i in range(50)
    )
    res_a = test_client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        data={"file": (io.BytesIO(csv_a), "dataset_A.csv", "text/csv"), "domain": "retail"},
    )
    assert res_a.status_code == 201
    ds_id_a = res_a.get_json()["data"]["dataset_id"]
    hash_a = res_a.get_json()["data"]["content_hash"]

    # Preprocess Dataset A
    prep_a = test_client.post(f"/api/v1/datasets/{ds_id_a}/preprocess", headers=headers)
    assert prep_a.status_code == 200

    # Forecast Dataset A (test camelCase datasetId parameter)
    fc_a_res = test_client.post(
        "/api/v1/forecasting/forecast",
        headers=headers,
        json={"domain": "retail", "datasetId": ds_id_a, "model": "random_forest", "n_estimators": 5},
    )
    assert fc_a_res.status_code == 200
    fc_a_data = fc_a_res.get_json()["data"]
    mean_pred_a = sum(p["predicted"] for p in fc_a_data["predictions"]) / len(fc_a_data["predictions"])
    assert fc_a_data["dataset_context"]["dataset_id"] == ds_id_a
    assert fc_a_data["dataset_context"]["dataset_hash"] == hash_a[:12]

    # 2. Prepare Dataset B (sales ~500,000)
    csv_b = b"Date,Sales,Promo,City\n" + b"\n".join(
        f"2026-01-{(i%28)+1:02d},{500000 + i * 1000},2,Delhi".encode("utf-8") for i in range(60)
    )
    res_b = test_client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        data={"file": (io.BytesIO(csv_b), "dataset_B.csv", "text/csv"), "domain": "retail"},
    )
    assert res_b.status_code == 201

    ds_id_b = res_b.get_json()["data"]["dataset_id"]
    hash_b = res_b.get_json()["data"]["content_hash"]

    assert ds_id_a != ds_id_b
    assert hash_a != hash_b

    # Preprocess Dataset B
    prep_b = test_client.post(f"/api/v1/datasets/{ds_id_b}/preprocess", headers=headers)
    assert prep_b.status_code == 200

    # Forecast Dataset B
    fc_b_res = test_client.post(
        "/api/v1/forecasting/forecast",
        headers=headers,
        json={"domain": "retail", "datasetId": ds_id_b, "model": "random_forest", "n_estimators": 5},
    )
    assert fc_b_res.status_code == 200
    fc_b_data = fc_b_res.get_json()["data"]
    mean_pred_b = sum(p["predicted"] for p in fc_b_data["predictions"]) / len(fc_b_data["predictions"])
    assert fc_b_data["dataset_context"]["dataset_id"] == ds_id_b
    assert fc_b_data["dataset_context"]["dataset_hash"] == hash_b[:12]
    assert abs(mean_pred_a - mean_pred_b) > 100000

    # 3. Switch back to Dataset A
    fc_a2_res = test_client.post(
        "/api/v1/forecasting/forecast",
        headers=headers,
        json={"domain": "retail", "datasetId": ds_id_a, "model": "random_forest", "n_estimators": 5},
    )
    assert fc_a2_res.status_code == 200
    fc_a2_data = fc_a2_res.get_json()["data"]
    mean_pred_a2 = sum(p["predicted"] for p in fc_a2_data["predictions"]) / len(fc_a2_data["predictions"])
    assert fc_a2_data["dataset_context"]["dataset_id"] == ds_id_a
    assert fc_a2_data["dataset_context"]["dataset_hash"] == hash_a[:12]
    assert abs(mean_pred_a - mean_pred_a2) < 1e-3
