import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app import create_app, db
from app.models.User import User
from app.services.dataset_service import upload_dataset, owned_dataset, dataset_detail
from app.services.preprocessing_service import preprocess_dataset
from app.services.ml_service import compare_service, forecast_service, explain_service

class AppConfig:
    TESTING = False
    SECRET_KEY = "audit-secret-key"
    JWT_SECRET_KEY = "audit-jwt-secret-key-32-bytes-long"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{ROOT / 'backend' / 'instance' / 'smartxsense.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]
    UPLOAD_FOLDER = str(ROOT / "storage" / "uploads")
    PROCESSED_DATA_FOLDER = str(ROOT / "storage" / "uploads" / "processed")
    FEATURES_DATA_FOLDER = str(ROOT / "ml_engine" / "data" / "features")
    ML_ARTIFACTS_ROOT = str(ROOT / "ml_engine" / "artifacts")

app = create_app(AppConfig)

with app.app_context():
    db.create_all()

    user = db.session.scalar(db.select(User).where(User.email == "e2e_two_dataset@smartxsense.io"))
    if not user:
        import uuid
        user = User(
            user_id=str(uuid.uuid4()),
            email="e2e_two_dataset@smartxsense.io",
            full_name="E2E Two Dataset Auditor",
            role="data_scientist",
            password_hash="hash"
        )
        db.session.add(user)
        db.session.commit()

    # DATASET A: Retail Daily Store Sales (Low scale: 100 to 500 units)
    dates_a = pd.date_range("2025-01-01", periods=100, freq="D")
    df_a = pd.DataFrame({
        "TransactionDate": dates_a.strftime("%Y-%m-%d"),
        "SalesRevenue": np.round(200.0 + np.arange(100) * 1.5, 2),
        "ItemQuantity": np.round(20 + np.arange(100) * 0.2).astype(int),
        "DiscountPct": np.random.choice([0.0, 0.05], 100),
    })
    # Add 5 exact duplicate rows to test duplicate removal
    df_a = pd.concat([df_a, df_a.iloc[:5]], ignore_index=True)
    bytes_a = df_a.to_csv(index=False).encode("utf-8")

    # DATASET B: High-Volume E-Commerce Orders (High scale: 50,000 to 100,000 INR)
    dates_b = pd.date_range("2025-06-01", periods=120, freq="D")
    df_b = pd.DataFrame({
        "OrderTimestamp": dates_b.strftime("%Y-%m-%d"),
        "GrandTotalINR": np.round(50000.0 + np.sin(np.arange(120) / 5.0) * 10000.0, 2),
        "OrderQuantity": np.round(500 + np.arange(120) * 2.0).astype(int),
    })
    bytes_b = df_b.to_csv(index=False).encode("utf-8")

    class CustomFile:
        def __init__(self, filename, content):
            self.filename = filename
            self.mimetype = "text/csv"
            self.content = content
        def read(self):
            return self.content
        def save(self, dst):
            Path(dst).write_bytes(self.content)

    print("==================================================")
    print("STEP 1: UPLOAD DATASET A (store_sales_low_scale.csv)")
    print("==================================================")
    up_a = upload_dataset(CustomFile("store_sales_low_scale.csv", bytes_a), domain="retail", user_id=user.user_id)
    id_a = up_a.dataset_id if hasattr(up_a, "dataset_id") else up_a["dataset_id"]
    hash_a = up_a.content_hash if hasattr(up_a, "content_hash") else up_a["content_hash"]
    print(f"Dataset A ID:   {id_a}")
    print(f"Dataset A Hash: {hash_a}")

    def get_pred_val(p_item):
        for k in ("predicted_sales", "predicted_demand_mw", "predicted", "forecast"):
            if k in p_item:
                return float(p_item[k])
        return float(list(p_item.values())[1]) if len(p_item) > 1 else 0.0

    # Train and Forecast on Dataset A
    fc_a1 = forecast_service({"domain": "retail", "datasetId": id_a, "model": "random_forest"})
    m_a1 = fc_a1["test_metrics"]
    pred_a1 = get_pred_val(fc_a1["predictions"][0])
    print(f"Dataset A (Run 1) Test RMSE: {m_a1['rmse']:.4f} | First Forecast Pred: {pred_a1:.2f}")

    print("\n==================================================")
    print("STEP 2: UPLOAD DATASET B (ecommerce_high_scale.csv)")
    print("==================================================")
    up_b = upload_dataset(CustomFile("ecommerce_high_scale.csv", bytes_b), domain="retail", user_id=user.user_id)
    id_b = up_b.dataset_id if hasattr(up_b, "dataset_id") else up_b["dataset_id"]
    hash_b = up_b.content_hash if hasattr(up_b, "content_hash") else up_b["content_hash"]
    print(f"Dataset B ID:   {id_b}")
    print(f"Dataset B Hash: {hash_b}")

    # Train and Forecast on Dataset B
    fc_b = forecast_service({"domain": "retail", "datasetId": id_b, "model": "random_forest"})
    m_b = fc_b["test_metrics"]
    pred_b = get_pred_val(fc_b["predictions"][0])
    print(f"Dataset B Test RMSE: {m_b['rmse']:.4f} | First Forecast Pred: {pred_b:.2f}")

    # Verify predictions and scales are completely different between Dataset A and Dataset B
    assert abs(pred_a1 - pred_b) > 100.0, f"Error: Dataset A and Dataset B predictions are too close! ({pred_a1} vs {pred_b})"
    print("[SUCCESS] Dataset A vs Dataset B Output Isolation Verified! (Predictions scale: A ~ 340 vs B ~ 50,000)")

    print("\n==================================================")
    print("STEP 3: SWITCH BACK TO DATASET A (A -> B -> A Isolation Test)")
    print("==================================================")
    fc_a2 = forecast_service({"domain": "retail", "datasetId": id_a, "model": "random_forest"})
    m_a2 = fc_a2["test_metrics"]
    pred_a2 = get_pred_val(fc_a2["predictions"][0])
    is_reused_a2 = fc_a2.get("execution", {}).get("artifact_reused", False)

    print(f"Dataset A (Run 2) Test RMSE: {m_a2['rmse']:.4f} | First Forecast Pred: {pred_a2:.2f}")
    print(f"Dataset A Artifact Reused: {is_reused_a2}")

    assert pred_a1 == pred_a2, "Error: Returning to Dataset A produced inconsistent prediction!"
    assert is_reused_a2 is True, "Error: Returning to Dataset A did not reuse Dataset A's own artifact!"

    print("\n[SUCCESS] TWO-DATASET A -> B -> A ISOLATION & ARTIFACT REUSE VERIFIED 100%!")


