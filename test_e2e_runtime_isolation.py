import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app import create_app, db
from app.models.User import User
from app.services.dataset_service import upload_dataset
from app.services.ml_service import compare_service, explain_service, forecast_service

tmp = ROOT / "test_e2e_tmp"
tmp.mkdir(exist_ok=True)

class TestCfg:
    TESTING = True
    SECRET_KEY = "test"
    JWT_SECRET_KEY = "jwt-secret-at-least-32-bytes-long"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp / 'e2e_test.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]
    ML_ARTIFACTS_ROOT = str(tmp / "artifacts")
    UPLOAD_FOLDER = str(tmp / "uploads")
    PROCESSED_DATA_FOLDER = str(tmp / "processed")
    FEATURES_DATA_FOLDER = str(tmp / "features")

app = create_app(TestCfg)

with app.app_context():
    db.create_all()
    
    import uuid
    user_id = str(uuid.uuid4())
    user = User(user_id=user_id, email=f"test_e2e_{user_id[:8]}@example.com", full_name="Test User", role="data_scientist", password_hash="test_hash")

    db.session.add(user)
    db.session.commit()



    # ----------------------------------------------------
    # DATASET A: Retail dataset with normal scale (~1,000)
    # ----------------------------------------------------
    np.random.seed(42)
    dates_a = pd.date_range("2026-01-01", periods=100, freq="D")
    sales_a = np.random.uniform(1000, 2000, size=100)
    df_a = pd.DataFrame({
        "Date": dates_a,
        "Sales": sales_a,
        "Discount": np.random.uniform(0, 0.2, size=100),
        "Store_City": ["Mumbai"] * 50 + ["Delhi"] * 50,
    })
    path_a = tmp / "dataset_A.csv"
    df_a.to_csv(path_a, index=False)

    class MockFileA:
        filename = "dataset_A.csv"
        mimetype = "text/csv"
        def read(self):
            return path_a.read_bytes()
        def save(self, dst):
            Path(dst).write_bytes(path_a.read_bytes())


    res_a = upload_dataset(MockFileA(), domain="retail", user_id=user_id)

    ds_id_a = getattr(res_a, "dataset_id", None) or (res_a.get("dataset_id") if isinstance(res_a, dict) else None)
    fn_a = getattr(res_a, "filename", None) or (res_a.get("filename") if isinstance(res_a, dict) else None)
    hash_a = getattr(res_a, "content_hash", None) or (res_a.get("content_hash") if isinstance(res_a, dict) else None)

    print("--- DATASET A RECORD ---")
    print(f"  dataset_id: {ds_id_a}")
    print(f"  filename: {fn_a}")
    print(f"  content_hash: {hash_a}")
    print(f"  row_count: {len(df_a)}, col_count: {len(df_a.columns)}")
    print(f"  detected target: Sales -> sales")

    # Forecast Dataset A
    fc_a = forecast_service({"domain": "retail", "datasetId": ds_id_a, "model": "random_forest"})
    mean_pred_a = np.mean([p["predicted"] for p in fc_a["predictions"]])
    metrics_a = fc_a["validation_metrics"]

    # Compare Dataset A
    comp_a = compare_service({"domain": "retail", "datasetId": ds_id_a, "models": ["random_forest", "xgboost"]})

    # Explain Dataset A
    exp_a = explain_service({"domain": "retail", "datasetId": ds_id_a, "model": "random_forest"})
    top_feat_a = exp_a["feature_importances"][0]["feature"] if exp_a.get("feature_importances") else None

    print(f"  Dataset A Forecast Mean Prediction: {mean_pred_a:.2f}")
    print(f"  Dataset A Metrics: {metrics_a}")
    print(f"  Dataset A Benchmark Winner: {comp_a['recommended_model']}")
    print(f"  Dataset A Top SHAP Feature: {top_feat_a}")
    print(f"  Dataset A Context in Response: {fc_a.get('dataset_context')}")
    print()

    # ----------------------------------------------------
    # DATASET B: Retail dataset with 100x scale (~500,000)
    # ----------------------------------------------------
    np.random.seed(99)
    dates_b = pd.date_range("2026-01-01", periods=120, freq="D")
    sales_b = np.random.uniform(500000, 900000, size=120)
    df_b = pd.DataFrame({
        "Date": dates_b,
        "Sales": sales_b,
        "Promotion_Level": np.random.uniform(1, 5, size=120),
        "City": ["Bangalore"] * 60 + ["Chennai"] * 60,
    })
    path_b = tmp / "dataset_B.csv"
    df_b.to_csv(path_b, index=False)

    class MockFileB:
        filename = "dataset_B.csv"
        mimetype = "text/csv"
        def read(self):
            return path_b.read_bytes()
        def save(self, dst):
            Path(dst).write_bytes(path_b.read_bytes())

    res_b = upload_dataset(MockFileB(), domain="retail", user_id=user_id)

    ds_id_b = getattr(res_b, "dataset_id", None) or (res_b.get("dataset_id") if isinstance(res_b, dict) else None)
    fn_b = getattr(res_b, "filename", None) or (res_b.get("filename") if isinstance(res_b, dict) else None)
    hash_b = getattr(res_b, "content_hash", None) or (res_b.get("content_hash") if isinstance(res_b, dict) else None)

    print("--- DATASET B RECORD ---")
    print(f"  dataset_id: {ds_id_b}")
    print(f"  filename: {fn_b}")
    print(f"  content_hash: {hash_b}")
    print(f"  row_count: {len(df_b)}, col_count: {len(df_b.columns)}")


    # Forecast Dataset B
    fc_b = forecast_service({"domain": "retail", "datasetId": ds_id_b, "model": "random_forest"})
    mean_pred_b = np.mean([p["predicted"] for p in fc_b["predictions"]])
    metrics_b = fc_b["validation_metrics"]

    # Compare Dataset B
    comp_b = compare_service({"domain": "retail", "datasetId": ds_id_b, "models": ["random_forest", "xgboost"]})

    # Explain Dataset B
    exp_b = explain_service({"domain": "retail", "datasetId": ds_id_b, "model": "random_forest"})
    top_feat_b = exp_b["feature_importances"][0]["feature"] if exp_b.get("feature_importances") else None

    print(f"  Dataset B Forecast Mean Prediction: {mean_pred_b:.2f}")
    print(f"  Dataset B Metrics: {metrics_b}")
    print(f"  Dataset B Benchmark Winner: {comp_b['recommended_model']}")
    print(f"  Dataset B Top SHAP Feature: {top_feat_b}")
    print(f"  Dataset B Context in Response: {fc_b.get('dataset_context')}")
    print()

    # ----------------------------------------------------
    # VERIFICATION: Switch back to DATASET A
    # ----------------------------------------------------
    print("--- SWITCH BACK TO DATASET A ---")
    fc_a_restored = forecast_service({"domain": "retail", "datasetId": ds_id_a, "model": "random_forest"})
    mean_pred_a_restored = np.mean([p["predicted"] for p in fc_a_restored["predictions"]])

    print(f"  Dataset A Restored Forecast Mean Prediction: {mean_pred_a_restored:.2f}")
    print(f"  Dataset A Restored Context: {fc_a_restored.get('dataset_context')}")
    print()

    # ----------------------------------------------------
    # ISOLATION ASSERTIONS
    # ----------------------------------------------------
    assert ds_id_a != ds_id_b, "Dataset IDs must be distinct"
    assert hash_a != hash_b, "Content hashes must be distinct"
    assert abs(mean_pred_a - mean_pred_a_restored) < 1e-3, "Dataset A predictions must be perfectly restored"
    assert abs(mean_pred_a - mean_pred_b) > 100000, "Dataset A and Dataset B predictions must reflect their distinct magnitudes"
    assert fc_a["dataset_context"]["dataset_id"] == ds_id_a
    assert fc_b["dataset_context"]["dataset_id"] == ds_id_b
    assert fc_a_restored["dataset_context"]["dataset_id"] == ds_id_a
    print("[SUCCESS] E2E RUNTIME ISOLATION VERIFICATION PASSED PERFECTLY!")

