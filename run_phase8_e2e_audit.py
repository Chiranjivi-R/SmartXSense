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

    user = db.session.scalar(db.select(User).where(User.email == "phase8_auditor@smartxsense.io"))
    if not user:
        import uuid
        user = User(
            user_id=str(uuid.uuid4()),
            email="phase8_auditor@smartxsense.io",
            full_name="Phase 8 Auditor",
            role="data_scientist",
            password_hash="hash"
        )
        db.session.add(user)
        db.session.commit()

    # Generate a fresh custom retail dataset: new_retail_store.csv
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    df_raw = pd.DataFrame({
        "TransactionDate": dates.strftime("%Y-%m-%d"),
        "SalesRevenue": np.round(1500.0 + np.sin(np.arange(180) / 7.0) * 300.0 + np.random.normal(0, 20, 180), 2),
        "ItemQuantity": np.round(50 + np.sin(np.arange(180) / 7.0) * 10.0 + np.random.normal(0, 2, 180)).astype(int),
        "DiscountPct": np.random.choice([0.0, 0.05, 0.10, 0.15], 180),
        "Temperature_C": np.round(25.0 + np.cos(np.arange(180) / 30.0) * 5.0, 1),
        "CityName": np.random.choice(["Mumbai", "Delhi", "Bengaluru"], 180),
    })

    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")

    class FileObj:
        filename = "new_retail_store.csv"
        mimetype = "text/csv"
        def read(self):
            return csv_bytes
        def save(self, dst):
            Path(dst).write_bytes(csv_bytes)

    print("=== 1. REAL END-TO-END UPLOAD (new_retail_store.csv) ===")
    upload_res = upload_dataset(FileObj(), domain="retail", user_id=user.user_id)
    ds_id = upload_res.dataset_id if hasattr(upload_res, "dataset_id") else upload_res["dataset_id"]
    content_h = upload_res.content_hash if hasattr(upload_res, "content_hash") else upload_res["content_hash"]
    print(f"  dataset_id:   {ds_id}")
    print(f"  content_hash: {content_h}")

    print("\n=== 2. SEMANTIC INTELLIGENCE & SUITABILITY ANALYSIS ===")
    ds_obj = owned_dataset(ds_id, user.user_id)
    _, prep_res = preprocess_dataset(ds_obj, None)
    profile = dataset_detail(ds_obj, preview_rows=5)

    frame = pd.read_csv(Path(ds_obj.file_path))
    from ml_engine.preprocessing.intelligence import analyze_forecasting_suitability
    suitability = analyze_forecasting_suitability(frame, domain="retail")
    semantic_matches = suitability.get("semantic_matches", {})

    print(f"  Classification:       {suitability.get('classification')}")
    print(f"  Is Suitable:          {suitability.get('is_suitable')}")
    print(f"  Status Label:         {suitability.get('status_label')}")
    print(f"  Detected Date Col:    {suitability.get('detected_time_column')}")
    print(f"  Detected Target Col:  {suitability.get('detected_target_column')}")
    print(f"  Target Statistics:    {suitability.get('target_statistics')}")
    print("  Semantic Feature Matches:")
    for col, meta in semantic_matches.items():
        print(f"    - Column: {col:18s} | Family: {meta['matched_family']:20s} | Confidence: {meta['confidence_score']*100:3.0f}% | Reason: {meta['reason']}")

    print("\n=== 3. MODEL BENCHMARK & SELECTION (Validation-Driven) ===")
    comp_res = compare_service({
        "domain": "retail",
        "datasetId": ds_id,
        "models": ["random_forest", "xgboost", "lstm", "hybrid"],
        "selection_metric": "rmse"
    })

    rec_model = comp_res["recommended_model"]
    ranking = comp_res["ranking"]
    print(f"  Validation Winner: {rec_model.upper()}")
    print("  Benchmark Table:")
    print("  " + "-" * 105)
    print(f"  {'Rank':<5} {'Model':<15} {'Val RMSE':<10} {'Val MAE':<10} {'Val MAPE':<10} {'Val R2':<10} {'Test RMSE':<10} {'Selected':<8}")
    print("  " + "-" * 105)
    for r in ranking:
        v_m = r.get("validation_metrics", {})
        t_m = r.get("test_metrics", {})
        is_sel = "YES" if r["model"] == rec_model else "NO"
        print(f"  #{r.get('rank', '-'):<4} {r['model']:<15} {v_m.get('rmse', 0):<10.4f} {v_m.get('mae', 0):<10.4f} {v_m.get('mape', 0):<10.4f} {v_m.get('r2', 0):<10.4f} {t_m.get('rmse', 0):<10.4f} {is_sel:<8}")

    print("\n=== 4. TEST SET EVALUATION WITH SELECTED MODEL ===")
    fc_res = forecast_service({
        "domain": "retail",
        "datasetId": ds_id,
        "model": rec_model
    })

    val_metrics = fc_res.get("validation_metrics", {})
    test_metrics = fc_res.get("test_metrics", {})
    dataset_ctx = fc_res.get("dataset_context", {})
    execution = fc_res.get("execution", {})

    print(f"  Test Predictions Count: {fc_res.get('total_test_predictions')}")
    print(f"  Test RMSE:             {test_metrics.get('rmse'):.4f}")
    print(f"  Test MAE:              {test_metrics.get('mae'):.4f}")
    print(f"  Test MAPE:             {test_metrics.get('mape'):.4f}")
    print(f"  Test R2:               {test_metrics.get('r2'):.4f}")

    print("\n=== 5. SHAP EXPLAINABLE AI (XAI) ===")
    explain_res = explain_service({
        "domain": "retail",
        "datasetId": ds_id,
        "model": rec_model,
        "explanation_type": "global"
    })

    print(f"  Explainer Method:       {explain_res.get('explainer')}")
    print(f"  Natural Explanation:    {explain_res.get('natural_language_explanation')}")
    print(f"  Dataset Context Match:  dataset_id={explain_res.get('dataset_context', {}).get('dataset_id')}, hash={explain_res.get('dataset_context', {}).get('dataset_hash')}")
    print("  Top SHAP Features:")
    for feat in explain_res.get("features", [])[:5]:
        print(f"    - {feat.get('feature'):25s} | Importance: {feat.get('importance'):.4f}")

    print("\n[SUCCESS] REAL END-TO-END UPLOAD AUDIT COMPLETE!")
