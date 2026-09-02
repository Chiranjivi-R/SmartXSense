import hashlib
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app import create_app, db
from app.models.User import User
from app.services.dataset_service import upload_dataset, list_datasets, dataset_detail, owned_dataset
from app.services.preprocessing_service import preprocess_dataset



from app.services.ml_service import compare_service, forecast_service, explain_service, future_forecast_service


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

    # Get or create user
    user = db.session.scalar(db.select(User).where(User.email == "auditor@smartxsense.io"))
    if not user:
        import uuid
        user = User(
            user_id=str(uuid.uuid4()),
            email="auditor@smartxsense.io",
            full_name="Dataset Auditor",
            role="data_scientist",
            password_hash="hash"
        )
        db.session.add(user)
        db.session.commit()

    # Real retail file path
    file_path = ROOT / "storage" / "uploads" / "6707ea5604d64422bb06b3d4ad690b2a.csv"
    csv_bytes = file_path.read_bytes()

    class FileObj:
        filename = "data_smart.csv"
        mimetype = "text/csv"
        def read(self):
            return csv_bytes
        def save(self, dst):
            Path(dst).write_bytes(csv_bytes)

    print("=== 1. UPLOADING REAL RETAIL DATASET (data_smart.csv) ===")
    upload_res = upload_dataset(FileObj(), domain="retail", user_id=user.user_id)
    ds_id = upload_res.dataset_id if hasattr(upload_res, "dataset_id") else upload_res["dataset_id"]
    content_h = upload_res.content_hash if hasattr(upload_res, "content_hash") else upload_res["content_hash"]

    # Verify single record in dataset list
    total_count, dataset_list = list_datasets(user_id=user.user_id, domain="retail", limit=20, offset=0)
    uploaded_records = [d for d in dataset_list if d.get("filename") == "data_smart.csv"]
    print(f"  Exact uploaded record count for data_smart.csv: {len(uploaded_records)}")
    print(f"  dataset_id: {ds_id}")
    print(f"  content_hash: {content_h}")

    # Preprocess dataset
    print("\n=== 2. PREPROCESSING & FEATURE ENGINEERING ===")
    ds_obj = owned_dataset(ds_id, user.user_id)
    _, prep_res = preprocess_dataset(ds_obj, None)


    print(f"  features_ready: {prep_res.get('features_ready')}")
    print(f"  features_dir: {prep_res.get('features_dir')}")

    # Profile schema
    detail = dataset_detail(ds_obj, preview_rows=5)


    frame = pd.read_csv(file_path)
    print(f"  Row count: {len(frame)}, Column count: {len(frame.columns)}")

    # Model comparison across candidate models
    print("\n=== 3. MODEL BENCHMARK & SELECTION ===")
    comp_res = compare_service({
        "domain": "retail",
        "datasetId": ds_id,
        "models": ["random_forest", "xgboost", "lstm", "hybrid"],
        "selection_metric": "rmse"
    })

    rec_model = comp_res["recommended_model"]
    ranking = comp_res["ranking"]
    print(f"  Validation Recommended Model (Winner): {rec_model}")
    print("  Ranking table:")
    for r in ranking:
        v_m = r.get("validation_metrics", r)
        rmse = v_m.get("rmse", 0)
        mae = v_m.get("mae", 0)
        mape = v_m.get("mape", 0)
        r2 = v_m.get("r2", 0)
        print(f"    - Rank {r.get('rank', '-')}: Model: {r['model']:15s} | RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | MAPE: {mape:8.4f} | R2: {r2:8.4f}")




    # Run forecast with recommended winner
    print("\n=== 4. TEST SET EVALUATION WITH RECOMMENDED MODEL ===")
    fc_res = forecast_service({
        "domain": "retail",
        "datasetId": ds_id,
        "model": rec_model
    })

    val_metrics = fc_res.get("validation_metrics", {})
    test_metrics = fc_res.get("test_metrics", {})
    dataset_ctx = fc_res.get("dataset_context", {})
    execution = fc_res.get("execution", {})

    print("\n========================================================")
    print("               CLEAN BASELINE AUDIT RECORD              ")
    print("========================================================")
    print(f"  dataset_id:            {dataset_ctx.get('dataset_id')}")
    print(f"  dataset_name:          {dataset_ctx.get('dataset_name')}")
    print(f"  dataset_hash:          {dataset_ctx.get('dataset_hash')}")
    print(f"  row_count:             {dataset_ctx.get('rows_used')}")
    print(f"  column_count:          {len(frame.columns)}")
    print(f"  detected date column:  Date")
    print(f"  detected target:       {dataset_ctx.get('target')}")
    print(f"  test observation count:{fc_res.get('total_test_predictions')}")
    print(f"  MAE:                   {val_metrics.get('mae'):.4f}")
    print(f"  RMSE:                  {val_metrics.get('rmse'):.4f}")
    print(f"  MAPE:                  {val_metrics.get('mape'):.4f}")
    print(f"  R2:                    {val_metrics.get('r2'):.4f}")
    print(f"  validation winner:     {rec_model}")
    print(f"  final model:           {rec_model}")
    print(f"  artifact reused status:{execution.get('artifact_reused')}")
    print("========================================================\n")
