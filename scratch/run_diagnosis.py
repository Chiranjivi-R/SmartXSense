import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.cwd()))


import numpy as np
import pandas as pd

from ml_engine.models.trainer import (
    run_training_pipeline,
    evaluate_fitted_model_on_splits,
)
from ml_engine.evaluation.metrics import evaluate_forecast
from ml_engine.models.random_forest import RandomForestForecaster

features_dir = Path("ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4")
train_df = pd.read_parquet(features_dir / "train.parquet")
val_df = pd.read_parquet(features_dir / "validation.parquet")
test_df = pd.read_parquet(features_dir / "test.parquet")


print("=" * 80)
print("1. DATA USED FOR TRAINING")
print("=" * 80)
print(f"Dataset Feature Dir: {features_dir.name}")
print(f"Target Column: sales")
print(f"Time Column: timestamp")
print(f"Geographic Location: All Geographies (Demo Retail)")
print(f"Train Rows: {len(train_df)} | Date Range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"Val Rows:   {len(val_df)} | Date Range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"Test Rows:  {len(test_df)} | Date Range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
total_rows = len(train_df) + len(val_df) + len(test_df)
print(f"Total Rows across splits: {total_rows}")
print(f"Train/Val/Test Split Ratio: {len(train_df)/total_rows:.1%} / {len(val_df)/total_rows:.1%} / {len(test_df)/total_rows:.1%}")

# Check chronological ordering and no overlap
train_max = pd.to_datetime(train_df['timestamp']).max()
val_min = pd.to_datetime(val_df['timestamp']).min()
val_max = pd.to_datetime(val_df['timestamp']).max()
test_min = pd.to_datetime(test_df['timestamp']).min()
test_max = pd.to_datetime(test_df['timestamp']).max()

print(f"Strict Temporal Ordering Check:")
print(f"  Train Max ({train_max}) < Val Min ({val_min}): {train_max < val_min}")
print(f"  Val Max ({val_max}) < Test Min ({test_min}): {val_max < test_min}")
print(f"  Zero Leakage Confirmed: {train_max < val_min and val_max < test_min}")

feature_cols = [c for c in train_df.columns if c not in ("timestamp", "sales")]
print(f"\nFeatures in Feature Set ({len(feature_cols)}): {feature_cols}")

print("\n" + "=" * 80)
print("2. TRAINED RANDOM FOREST MODEL VERIFICATION & EVALUATION")
print("=" * 80)
t0 = time.perf_counter()
rf_res = run_training_pipeline(domain="retail", features_dir=features_dir, model_name="random_forest", force_retrain=True)
rf_train_time = time.perf_counter() - t0

print(f"Result keys: {list(rf_res.keys())}")
print(f"Model Name: {rf_res.get('model_name') or rf_res.get('model_info', {}).get('model_name')}")
print(f"Cache Key / Fingerprint: {rf_res.get('fingerprint') or rf_res.get('model_info', {}).get('dataset_fingerprint')}")
print(f"Hyperparameters: {rf_res.get('model_info', {}).get('hyperparameters', {})}")
print(f"Validation Metrics: {rf_res.get('val_metrics')}")
print(f"Test Metrics: {rf_res.get('test_metrics')}")
print(f"Baseline Metrics: {rf_res.get('baseline_metrics', {})}")
print(f"Baseline Improvement: {rf_res.get('baseline_comparison', {})}")


# 3. Error Analysis on Test Set
pred_df = pd.DataFrame(rf_res["predictions"])
y_test_true = pred_df["actual"].to_numpy(dtype=float)
y_test_pred = pred_df["predicted"].to_numpy(dtype=float)

abs_err = np.abs(y_test_true - y_test_pred)
sq_err = (y_test_true - y_test_pred) ** 2
pct_err = abs_err / np.where(y_test_true == 0, 1.0, np.abs(y_test_true)) * 100

error_df = pd.DataFrame({
    "timestamp": pred_df["timestamp"].values,
    "actual": y_test_true,
    "predicted": y_test_pred,
    "abs_error": abs_err,
    "sq_error": sq_err,
    "pct_error": pct_err,
})

print("\n" + "=" * 80)
print("3. DETAILED ERROR DISTRIBUTION & RMSE VS MAE ANALYSIS")
print("=" * 80)
mae = np.mean(abs_err)
rmse = np.sqrt(np.mean(sq_err))
med_ae = np.median(abs_err)
p75_ae = np.percentile(abs_err, 75)
p90_ae = np.percentile(abs_err, 90)
p95_ae = np.percentile(abs_err, 95)
p99_ae = np.percentile(abs_err, 99)
max_ae = np.max(abs_err)

print(f"MAE:               {mae:.4f}")
print(f"RMSE:              {rmse:.4f}")
print(f"RMSE / MAE Ratio:  {rmse / mae:.4f} (Theoretical Gaussian ratio is ~1.25; >1.5 indicates heavy-tailed errors)")
print(f"Median Abs Error:  {med_ae:.4f}")
print(f"75th Pct Abs Error:{p75_ae:.4f}")
print(f"90th Pct Abs Error:{p90_ae:.4f}")
print(f"95th Pct Abs Error:{p95_ae:.4f}")
print(f"99th Pct Abs Error:{p99_ae:.4f}")
print(f"Max Abs Error:     {max_ae:.4f}")

# Top 10 largest errors
top_errors = error_df.sort_values(by="abs_error", ascending=False).head(10)
print("\nTop 10 Largest Prediction Errors:")
print(top_errors.to_string(index=False))

# Calculate contribution of top 5% errors to total MSE (RMSE squared)
n_top_5pct = max(1, int(len(error_df) * 0.05))
top_5pct_df = error_df.sort_values(by="sq_error", ascending=False).head(n_top_5pct)
top_5pct_sq_sum = top_5pct_df['sq_error'].sum()
total_sq_sum = error_df['sq_error'].sum()
print(f"\nTop 5% worst observations ({n_top_5pct} rows out of {len(error_df)}) account for {top_5pct_sq_sum / total_sq_sum:.1%} of total squared error (RMSE driver)!")

print("\n" + "=" * 80)
print("4. MODEL COMPARISON ON EXACT SAME CHRONOLOGICAL SPLITS")
print("=" * 80)
models_to_test = ["naive", "arima", "random_forest", "xgboost", "lstm", "cnn_lstm", "hybrid"]
comparison_results = []

for m_name in models_to_test:
    try:
        t_start = time.perf_counter()
        m_res = run_training_pipeline(
            domain="retail",
            features_dir=features_dir,
            model_name=m_name,
            force_retrain=True,
        )
        t_elapsed = time.perf_counter() - t_start
        
        t_inf_start = time.perf_counter()
        _ = run_training_pipeline(domain="retail", features_dir=features_dir, model_name=m_name, force_retrain=False)
        t_inf = (time.perf_counter() - t_inf_start)
        
        tm = m_res.get("test_metrics", {})
        vm = m_res.get("validation_metrics", {})
        imp = m_res.get("improvement_pct", {})
        imp_val = imp if isinstance(imp, (int, float)) else (imp.get("mae") if isinstance(imp, dict) else None)
        
        comparison_results.append({
            "model": m_name,
            "test_mae": round(float(tm.get("mae", 0)), 3),
            "test_rmse": round(float(tm.get("rmse", 0)), 3),
            "test_mape": round(float(tm.get("mape", 0)) * 100, 2),
            "test_smape": round(float(tm.get("smape", 0)) * 100, 2),
            "test_wape": round(float(tm.get("wape", 0)) * 100, 2),
            "test_r2": round(float(tm.get("r2", 0)), 4),
            "val_mae": round(float(vm.get("mae", 0)), 3),
            "val_rmse": round(float(vm.get("rmse", 0)), 3),
            "val_mape": round(float(vm.get("mape", 0)) * 100, 2),
            "val_smape": round(float(vm.get("smape", 0)) * 100, 2),
            "val_wape": round(float(vm.get("wape", 0)) * 100, 2),
            "val_r2": round(float(vm.get("r2", 0)), 4),
            "improvement_vs_naive_pct": round(float(imp_val), 2) if imp_val is not None else None,
            "train_time_sec": round(t_elapsed, 3),
            "inf_time_sec": round(t_inf, 4),
        })
    except Exception as err:
        print(f"Model {m_name} failed: {err}")

comp_df = pd.DataFrame(comparison_results)
print(comp_df.to_string(index=False))

print("\n" + "=" * 80)
print("5. FEATURE IMPORTANCE & DRIFT DIAGNOSIS")
print("=" * 80)
# Feature statistics across train vs test
numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(train_df[f])]
categorical_features = [f for f in feature_cols if not pd.api.types.is_numeric_dtype(train_df[f])]

print(f"Numeric Features ({len(numeric_features)}):")
for f in numeric_features:
    train_mean, train_std = train_df[f].dropna().mean(), train_df[f].dropna().std()
    test_mean, test_std = test_df[f].dropna().mean(), test_df[f].dropna().std()
    valid_mask = train_df[f].notna() & train_df['sales'].notna()
    corr = np.corrcoef(train_df.loc[valid_mask, f], train_df.loc[valid_mask, 'sales'])[0, 1] if train_std > 0 else 0
    print(f"  Feature '{f}': Train Mean={train_mean:.2f} (std={train_std:.2f}) | Test Mean={test_mean:.2f} (std={test_std:.2f}) | Corr with Sales={corr:.3f}")

print(f"\nCategorical Features ({len(categorical_features)}):")
for f in categorical_features:
    train_n_unique = train_df[f].nunique()
    test_n_unique = test_df[f].nunique()
    print(f"  Feature '{f}': Train Unique={train_n_unique}, Test Unique={test_n_unique}")

# Target stats
print(f"\nTarget 'sales' Distribution:")
print(f"  Train: Mean={train_df['sales'].mean():.2f}, Std={train_df['sales'].std():.2f}, Median={train_df['sales'].median():.2f}, Min={train_df['sales'].min():.2f}, Max={train_df['sales'].max():.2f}")
print(f"  Val:   Mean={val_df['sales'].mean():.2f}, Std={val_df['sales'].std():.2f}, Median={val_df['sales'].median():.2f}, Min={val_df['sales'].min():.2f}, Max={val_df['sales'].max():.2f}")
print(f"  Test:  Mean={test_df['sales'].mean():.2f}, Std={test_df['sales'].std():.2f}, Median={test_df['sales'].median():.2f}, Min={test_df['sales'].min():.2f}, Max={test_df['sales'].max():.2f}")

