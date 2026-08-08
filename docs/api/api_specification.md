# SmartXSense — REST API Contract Specification
**Explainable Hybrid Demand Forecasting Framework for Retail and Electricity Systems**

> **SUPERSEDED — do not use as the live API contract.**
> This document is a legacy draft and describes many endpoints and behaviors that are **not** implemented (async training jobs, domain-split train/predict routes, PDF reports, mutable settings, fabricated dashboard KPIs, etc.).
> **Authoritative sources:** [`docs/ML_API_CONTRACT.md`](../ML_API_CONTRACT.md), [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), [`docs/MODEL_CAPABILITY_MATRIX.md`](../MODEL_CAPABILITY_MATRIX.md), and [`README.md`](../../README.md).

---

## 1. Overview & API Architecture

~~This specification defines the complete RESTful API contract for the SmartXSense backend.~~ *(Legacy draft retained for historical reference only.)* All API requests and responses utilize standard JSON format (except binary file uploads and downloads).

* **Base URL**: `http://localhost:5000/api/v1`
* **Authentication Standard**: JSON Web Token (JWT) passed in the `Authorization: Bearer <token>` HTTP header.
* **Global Error Format**:
```json
{
  "status": "error",
  "code": 400,
  "message": "Human-readable error explanation",
  "errors": { "field_name": ["Specific validation error"] }
}
```

---

## 2. API Endpoints Specification

### 2.1 Authentication Module

#### `POST /api/v1/auth/register`
* **Purpose**: Register a new user account.
* **Authentication Required**: No
* **Frontend Page**: `AuthPage.jsx` / `LoginPage.jsx`
* **Backend Module**: `backend/app/api/auth/` (`auth_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "email": "analyst@smartxsense.ai",
  "password": "SecurePassword123!",
  "full_name": "Data Analyst",
  "role": "analyst"
}
```
* **Success Response (201 Created)**:
```json
{
  "status": "success",
  "code": 201,
  "message": "User registered successfully",
  "data": {
    "user_id": "usr_94821a",
    "email": "analyst@smartxsense.ai",
    "full_name": "Data Analyst",
    "role": "analyst",
    "created_at": "2026-07-30T17:00:00Z"
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Missing mandatory fields or weak password format.
  * **409 Conflict**: Email already registered.

---

#### `POST /api/v1/auth/login`
* **Purpose**: Authenticate user credentials and return JWT access/refresh tokens.
* **Authentication Required**: No
* **Frontend Page**: `LoginPage.jsx`
* **Backend Module**: `backend/app/api/auth/` (`auth_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "email": "analyst@smartxsense.ai",
  "password": "SecurePassword123!"
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_in": 3600,
    "user": {
      "user_id": "usr_94821a",
      "email": "analyst@smartxsense.ai",
      "full_name": "Data Analyst",
      "role": "analyst"
    }
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Invalid email or password credentials.

---

#### `POST /api/v1/auth/refresh`
* **Purpose**: Refresh an expired JWT access token using a valid refresh token.
* **Authentication Required**: Yes (Refresh Token)
* **Frontend Page**: Shared `apiClient.js` Interceptor
* **Backend Module**: `backend/app/api/auth/` (`auth_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_in": 3600
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Invalid or expired refresh token.

---

#### `GET /api/v1/auth/me`
* **Purpose**: Retrieve current logged-in user profile details.
* **Authentication Required**: Yes
* **Frontend Page**: `SettingsPage.jsx` / TopBar Navigation
* **Backend Module**: `backend/app/api/auth/` (`auth_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "user_id": "usr_94821a",
    "email": "analyst@smartxsense.ai",
    "full_name": "Data Analyst",
    "role": "analyst",
    "preferences": {
      "theme": "dark",
      "notifications": true
    }
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Missing or revoked Bearer token.

---

### 2.2 Dataset Upload & Management Module

#### `POST /api/v1/datasets/upload`
* **Purpose**: Ingest raw CSV or XLSX dataset files for retail or electricity domain processing.
* **Authentication Required**: Yes
* **Frontend Page**: `UploadPage.jsx` / `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: `ml_engine/data/raw/` file staging
* **Request Parameters**: Form Data (`multipart/form-data`)
  * `file`: File binary (`.csv`, `.xlsx`)
  * `domain`: Query parameter (`retail` | `electricity`)
  * `description`: Text string
* **Request JSON Body**: N/A (Multipart form-data)
* **Success Response (201 Created)**:
```json
{
  "status": "success",
  "code": 201,
  "message": "Dataset uploaded successfully",
  "data": {
    "dataset_id": "ds_retail_8820",
    "filename": "retail_sales_2026.csv",
    "domain": "retail",
    "file_size_bytes": 4829100,
    "row_count": 52400,
    "column_count": 14,
    "status": "uploaded",
    "upload_timestamp": "2026-07-30T17:05:00Z"
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Unsupported file extension or file size exceeds maximum limit.
  * **415 Unsupported Media Type**: Invalid content-type.

---

#### `GET /api/v1/datasets`
* **Purpose**: List all uploaded datasets filtered by domain and upload status.
* **Authentication Required**: Yes
* **Frontend Page**: `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**:
  * `domain` (optional): `retail` | `electricity`
  * `limit` (optional): integer (default `20`)
  * `offset` (optional): integer (default `0`)
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "total": 1,
    "datasets": [
      {
        "dataset_id": "ds_retail_8820",
        "filename": "retail_sales_2026.csv",
        "domain": "retail",
        "row_count": 52400,
        "status": "validated",
        "created_at": "2026-07-30T17:05:00Z"
      }
    ]
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Missing token.

---

#### `GET /api/v1/datasets/{dataset_id}`
* **Purpose**: Fetch dataset details, metadata summary, and top N rows preview.
* **Authentication Required**: Yes
* **Frontend Page**: `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**:
  * `dataset_id` (Path): String
  * `preview_rows` (Query, optional): Integer (default `10`)
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "dataset_id": "ds_retail_8820",
    "filename": "retail_sales_2026.csv",
    "domain": "retail",
    "columns": ["timestamp", "store_id", "item_id", "sales", "promo", "temperature"],
    "preview": [
      { "timestamp": "2026-01-01", "store_id": "S01", "item_id": "I101", "sales": 420, "promo": 1, "temperature": 24.5 }
    ]
  }
}
```
* **Error Responses**:
  * **404 Not Found**: Dataset ID does not exist.

---

#### `DELETE /api/v1/datasets/{dataset_id}`
* **Purpose**: Delete a dataset and its associated processed files.
* **Authentication Required**: Yes (Admin / Analyst)
* **Frontend Page**: `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: Storage cleanup
* **Request Parameters**:
  * `dataset_id` (Path): String
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "message": "Dataset ds_retail_8820 deleted successfully"
}
```
* **Error Responses**:
  * **404 Not Found**: Dataset ID not found.

---

### 2.3 Data Validation & Preprocessing Module

#### `POST /api/v1/datasets/{dataset_id}/validate`
* **Purpose**: Perform automated schema validation, missing value audit, and data type checks.
* **Authentication Required**: Yes
* **Frontend Page**: `UploadPage.jsx` / `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: `ml_engine/preprocessing/validator.py`
* **Request Parameters**:
  * `dataset_id` (Path): String
* **Request JSON Body**:
```json
{
  "schema_type": "retail_v1",
  "strict_mode": true
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "dataset_id": "ds_retail_8820",
    "is_valid": true,
    "missing_value_summary": {
      "sales": 12,
      "temperature": 0
    },
    "duplicate_rows": 0,
    "validation_logs": ["Schema check passed", "12 missing sales values flagged for imputation"]
  }
}
```
* **Error Responses**:
  * **422 Unprocessable Entity**: Critical schema failure (e.g. missing mandatory `timestamp` column).

---

#### `POST /api/v1/datasets/{dataset_id}/preprocess`
* **Purpose**: Execute outlier removal, missing value imputation, scaling, and lag/cyclical feature generation.
* **Authentication Required**: Yes
* **Frontend Page**: `TrainingPage.jsx` / `DatasetManagementPage.jsx`
* **Backend Module**: `backend/app/api/datasets/` (`dataset_service.py`)
* **ML Module Invoked**: `ml_engine/preprocessing/cleaner.py` & `ml_engine/feature_engineering/`
* **Request Parameters**:
  * `dataset_id` (Path): String
* **Request JSON Body**:
```json
{
  "imputation_strategy": "interpolate",
  "outlier_method": "iqr",
  "scaler": "standard",
  "generate_lags": true,
  "lag_intervals": [1, 7, 14, 30]
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "message": "Preprocessing and feature engineering completed",
  "data": {
    "processed_dataset_id": "proc_retail_8820",
    "processed_rows": 52388,
    "features_generated": ["sales_lag_1", "sales_lag_7", "rolling_mean_7", "day_of_week_sin", "day_of_week_cos"],
    "output_file_path": "ml_engine/data/processed/retail/proc_retail_8820.parquet"
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid preprocessing parameters.

---

### 2.4 Model Training & Evaluation Module

#### `POST /api/v1/forecasting/train/retail`
* **Purpose**: Trigger background model training for retail demand (ARIMA, XGBoost, LSTM).
* **Authentication Required**: Yes
* **Frontend Page**: `TrainingPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/pipelines/training_pipeline.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "processed_dataset_id": "proc_retail_8820",
  "target_column": "sales",
  "selected_models": ["xgboost", "lstm"],
  "hyperparameters": {
    "xgboost": { "n_estimators": 200, "learning_rate": 0.05 },
    "lstm": { "epochs": 50, "batch_size": 32, "units": 64 }
  },
  "test_split_ratio": 0.2
}
```
* **Success Response (202 Accepted)**:
```json
{
  "status": "success",
  "code": 202,
  "message": "Retail model training job initiated",
  "data": {
    "job_id": "job_tr_retail_9941",
    "status": "in_progress",
    "estimated_duration_seconds": 45
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Dataset missing or invalid hyperparameter key.

---

#### `POST /api/v1/forecasting/train/electricity`
* **Purpose**: Trigger model training for electricity demand (SARIMAX, CNN-LSTM).
* **Authentication Required**: Yes
* **Frontend Page**: `TrainingPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/pipelines/training_pipeline.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "processed_dataset_id": "proc_elec_4120",
  "target_column": "grid_load_mw",
  "selected_models": ["sarimax", "cnn_lstm"],
  "hyperparameters": {
    "sarimax": { "order": [2, 1, 2], "seasonal_order": [1, 1, 1, 24] },
    "cnn_lstm": { "filters": 32, "kernel_size": 3, "lstm_units": 50, "epochs": 40 }
  }
}
```
* **Success Response (202 Accepted)**:
```json
{
  "status": "success",
  "code": 202,
  "message": "Electricity model training job initiated",
  "data": {
    "job_id": "job_tr_elec_1022",
    "status": "in_progress"
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid training payload.

---

#### `GET /api/v1/forecasting/models`
* **Purpose**: Retrieve registry list of all trained model versions and saved artifacts.
* **Authentication Required**: Yes
* **Frontend Page**: `ModelsPage.jsx` / `TrainingPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/artifacts/metadata/`
* **Request Parameters**:
  * `domain` (Query, optional): `retail` | `electricity`
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "models": [
      {
        "model_id": "mdl_retail_xgb_v1",
        "domain": "retail",
        "algorithm": "XGBoost",
        "accuracy": 95.2,
        "mape": 4.8,
        "rmse": 196,
        "created_at": "2026-07-30T15:30:00Z",
        "artifact_path": "ml_engine/artifacts/models/xgboost_retail_v1.pkl"
      }
    ]
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Missing token.

---

#### `GET /api/v1/forecasting/models/{model_id}/evaluation`
* **Purpose**: Fetch detailed out-of-sample backtesting metrics (MAE, RMSE, MAPE, WAPE, R²).
* **Authentication Required**: Yes
* **Frontend Page**: `ModelsPage.jsx` / `ForecastPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/evaluation/evaluator.py`
* **Request Parameters**:
  * `model_id` (Path): String
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "model_id": "mdl_retail_hybrid_v1",
    "algorithm": "Hybrid (Statistical + LSTM)",
    "metrics": {
      "mae": 124.5,
      "rmse": 196.2,
      "mape": 2.9,
      "wape": 2.7,
      "r2_score": 0.982
    },
    "cross_validation_folds": 5,
    "backtest_plot_data": [
      { "timestamp": "2026-07-01", "actual": 4200, "predicted": 4150 }
    ]
  }
}
```
* **Error Responses**:
  * **404 Not Found**: Model ID not found.

---

### 2.5 Prediction & Hybrid Forecast Engine Module

#### `POST /api/v1/forecasting/predict/retail`
* **Purpose**: Generate demand predictions for retail datasets.
* **Authentication Required**: Yes
* **Frontend Page**: `RetailPage.jsx` / `ForecastPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/pipelines/inference_pipeline.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "model_id": "mdl_retail_xgb_v1",
  "forecast_horizon_days": 7,
  "confidence_level": 95,
  "exogenous_variables": {
    "temperature": [28.5, 29.0, 31.2, 30.5, 27.8, 26.5, 28.0],
    "promotion": [1, 1, 0, 0, 0, 1, 1]
  }
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "forecast_id": "fc_ret_88301",
    "model_id": "mdl_retail_xgb_v1",
    "domain": "retail",
    "predictions": [
      { "date": "2026-07-31", "predicted_sales": 4100, "lower_ci": 3900, "upper_ci": 4300 },
      { "date": "2026-08-01", "predicted_sales": 7650, "lower_ci": 7400, "upper_ci": 7900 }
    ]
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid horizon or missing exogenous features.
  * **404 Not Found**: Model ID not found.

---

#### `POST /api/v1/forecasting/predict/electricity`
* **Purpose**: Generate demand predictions for electricity datasets.
* **Authentication Required**: Yes
* **Frontend Page**: `ElectricityPage.jsx` / `ForecastPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/pipelines/inference_pipeline.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "model_id": "mdl_elec_cnn_lstm_v1",
  "forecast_horizon_hours": 24,
  "confidence_level": 95,
  "grid_region": "West",
  "exogenous_variables": {
    "ambient_temperature": [32.0, 31.5, 30.2, 29.0],
    "humidity": [65, 68, 70, 72]
  }
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "forecast_id": "fc_elec_4021",
    "model_id": "mdl_elec_cnn_lstm_v1",
    "domain": "electricity",
    "grid_region": "West",
    "predictions": [
      { "hour": "00:00", "predicted_load_mw": 2050, "lower_ci": 1950, "upper_ci": 2150 },
      { "hour": "03:00", "predicted_load_mw": 1850, "lower_ci": 1750, "upper_ci": 1950 }
    ]
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid region or missing hourly features.
  * **404 Not Found**: Model ID not found.

---

#### `POST /api/v1/forecasting/hybrid-forecast`
* **Purpose**: Run the hybrid forecasting engine combining statistical baseline predictions with non-linear LSTM residual corrections.
* **Authentication Required**: Yes
* **Frontend Page**: `RetailPage.jsx` / `ElectricityPage.jsx` / `DashboardPage.jsx`
* **Backend Module**: `backend/app/api/forecasting/` (`forecast_service.py`)
* **ML Module Invoked**: `ml_engine/hybrid_engine/hybrid_forecaster.py` & `ensemble.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "domain": "retail",
  "statistical_model": "arima",
  "deep_model": "lstm",
  "horizon": 7,
  "blending_weight": 0.6
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "engine": "SmartXSense Hybrid Engine v1",
    "statistical_baseline_mape": 8.4,
    "hybrid_final_mape": 2.9,
    "combined_forecast": [
      { "timestamp": "2026-07-31", "statistical_base": 3900, "lstm_residual": 200, "hybrid_output": 4100 }
    ]
  }
}
```
* **Error Responses**:
  * **500 Internal Server Error**: Ensemble blending calculation failure.

---

### 2.6 Explainable AI (XAI) Module

#### `POST /api/v1/explainability/shap`
* **Purpose**: Compute SHAP global feature importances and local waterfall attribution values.
* **Authentication Required**: Yes
* **Frontend Page**: `XAIPage.jsx`
* **Backend Module**: `backend/app/api/explainability/` (`xai_service.py`)
* **ML Module Invoked**: `ml_engine/explainability/shap_explainer.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "model_id": "mdl_retail_xgb_v1",
  "instance_data": {
    "day_of_week": "Saturday",
    "temperature": 32.5,
    "promo_active": 1,
    "prev_week_sales": 7200
  }
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "base_value": 4500,
    "prediction_value": 7800,
    "shap_values": [
      { "feature": "Day of Week", "value": "Saturday", "attribution": 1950, "direction": "positive" },
      { "feature": "Temperature", "value": 32.5, "attribution": 850, "direction": "positive" },
      { "feature": "Promotion Active", "value": 1, "attribution": 500, "direction": "positive" }
    ]
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Missing feature attributes in instance payload.

---

#### `POST /api/v1/explainability/lime`
* **Purpose**: Compute LIME tabular local explanations for instance-level prediction breakdowns.
* **Authentication Required**: Yes
* **Frontend Page**: `XAIPage.jsx`
* **Backend Module**: `backend/app/api/explainability/` (`xai_service.py`)
* **ML Module Invoked**: `ml_engine/explainability/lime_explainer.py`
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "model_id": "mdl_elec_lstm_v1",
  "num_features": 5,
  "sample_instance": {
    "hour": 18,
    "temperature": 38,
    "working_day": 1
  }
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "intercept": 5200,
    "local_prediction": 8100,
    "explanation_rules": [
      { "rule": "hour > 17.00", "weight": 0.35 },
      { "rule": "temperature > 35.00", "weight": 0.18 }
    ]
  }
}
```
* **Error Responses**:
  * **500 Internal Server Error**: LIME explainer generation error.

---

### 2.7 Dashboard Module

#### `GET /api/v1/dashboard/overview`
* **Purpose**: Return overall dashboard KPIs, summary statistics, and system health status.
* **Authentication Required**: Yes
* **Frontend Page**: `DashboardPage.jsx`
* **Backend Module**: `backend/app/api/dashboard/` (`dashboard_routes.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "retail_accuracy": 97.1,
    "energy_mape": 2.9,
    "skus_tracked": 14280,
    "grid_regions": 5,
    "system_health": "optimal",
    "recent_alerts": [
      { "id": 1, "type": "alert", "message": "Peak electricity demand forecasted for 18:00 today", "time": "2m ago" }
    ]
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Token missing or invalid.

---

#### `GET /api/v1/dashboard/retail`
* **Purpose**: Return retail dashboard charts, category growth rates, KPIs, and forecasting metrics.
* **Authentication Required**: Yes
* **Frontend Page**: `RetailPage.jsx` / `DashboardPage.jsx`
* **Backend Module**: `backend/app/api/dashboard/` (`dashboard_routes.py`)
* **ML Module Invoked**: None
* **Request Parameters**:
  * `period` (Query, optional): `daily` | `weekly` | `monthly` (default `weekly`)
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "monthly_revenue": "₹91.2L",
    "forecast_accuracy": 97.1,
    "category_growth": [
      { "name": "Electronics", "value": 35, "growth": 12.4 },
      { "name": "Apparel", "value": 22, "growth": 8.1 }
    ],
    "sales_vs_forecast": [
      { "month": "Jan", "sales": 42000, "inventory": 38000, "forecast": 44000 }
    ]
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Token missing or invalid.

---

#### `GET /api/v1/dashboard/electricity`
* **Purpose**: Return electricity dashboard charts, regional grid load, KPIs, and forecasting metrics.
* **Authentication Required**: Yes
* **Frontend Page**: `ElectricityPage.jsx` / `DashboardPage.jsx`
* **Backend Module**: `backend/app/api/dashboard/` (`dashboard_routes.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "current_load_mw": 7840,
    "peak_today_gw": 10.2,
    "renewable_share": 38.4,
    "grid_mape": 2.9,
    "regional_demand": [
      { "region": "North", "peak": 8200, "avg": 5400, "offpeak": 2100 }
    ]
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Token missing or invalid.

---

### 2.8 Reports Generation Module

#### `POST /api/v1/reports/generate`
* **Purpose**: Render an executive analytical report in PDF or CSV format containing forecast tables and SHAP plots.
* **Authentication Required**: Yes
* **Frontend Page**: `ReportsPage.jsx` / `RetailPage.jsx`
* **Backend Module**: `backend/app/api/reports/` (`report_service.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "report_type": "retail_forecast",
  "format": "pdf",
  "start_date": "2026-07-01",
  "end_date": "2026-07-30",
  "include_xai_plots": true
}
```
* **Success Response (201 Created)**:
```json
{
  "status": "success",
  "code": 201,
  "data": {
    "report_id": "rep_pdf_9012",
    "filename": "smartxsense_retail_report_20260730.pdf",
    "download_url": "/api/v1/reports/rep_pdf_9012/download",
    "created_at": "2026-07-30T17:10:00Z"
  }
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid date range or unsupported format.

---

#### `GET /api/v1/reports/{report_id}/download`
* **Purpose**: Download generated PDF or CSV analytics report file.
* **Authentication Required**: Yes
* **Frontend Page**: `ReportsPage.jsx`
* **Backend Module**: `backend/app/api/reports/` (`report_service.py`)
* **ML Module Invoked**: Storage stream
* **Request Parameters**:
  * `report_id` (Path): String
* **Request JSON Body**: None
* **Success Response (200 OK)**: Binary file stream (`application/pdf` or `text/csv`).
* **Error Responses**:
  * **404 Not Found**: Report file expired or not found.

---

### 2.9 Settings Module

#### `GET /api/v1/settings`
* **Purpose**: Fetch platform configurations, forecast horizons, and threshold preferences.
* **Authentication Required**: Yes
* **Frontend Page**: `SettingsPage.jsx`
* **Backend Module**: `backend/app/api/settings/` (`settings_routes.py`)
* **ML Module Invoked**: None
* **Request Parameters**: None
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "notifications_enabled": true,
    "dark_mode": true,
    "auto_retrain_days": 30,
    "forecast_horizon_days": 30,
    "confidence_level": 95,
    "update_frequency": "hourly"
  }
}
```
* **Error Responses**:
  * **401 Unauthorized**: Unauthenticated request.

---

#### `PUT /api/v1/settings`
* **Purpose**: Update system configuration parameters and forecast settings.
* **Authentication Required**: Yes
* **Frontend Page**: `SettingsPage.jsx`
* **Backend Module**: `backend/app/api/settings/` (`settings_routes.py`)
* **ML Module Invoked**: Config YAML update
* **Request Parameters**: None
* **Request JSON Body**:
```json
{
  "notifications_enabled": true,
  "auto_retrain_days": 30,
  "forecast_horizon_days": 30,
  "confidence_level": 95,
  "update_frequency": "hourly"
}
```
* **Success Response (200 OK)**:
```json
{
  "status": "success",
  "code": 200,
  "message": "System settings updated successfully"
}
```
* **Error Responses**:
  * **400 Bad Request**: Invalid horizon range (must be between 7 and 90 days).

---

### 2.10 Health Check Module

#### `GET /api/v1/health`
* **Purpose**: System health check endpoint inspecting database connectivity, ML engine status, and storage access.
* **Authentication Required**: No
* **Frontend Page**: Infrastructure / Monitoring
* **Backend Module**: `backend/app/api/health/` (`health_routes.py`)
* **ML Module Invoked**: Engine status check
* **Request Parameters**: None
* **Request JSON Body**: None
* **Success Response (200 OK)**:
```json
{
  "status": "healthy",
  "code": 200,
  "timestamp": "2026-07-30T17:15:00Z",
  "components": {
    "database": "connected",
    "ml_engine": "ready",
    "storage": "writable"
  }
}
```
* **Error Responses**:
  * **503 Service Unavailable**: Database or ML engine unresponsive.

---

## 3. Summary Table: REST API Matrix

| Method | Endpoint URL | Frontend Page | Backend Module | ML Module |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | `LoginPage.jsx` | `backend/app/api/auth/` | N/A |
| `POST` | `/api/v1/auth/login` | `LoginPage.jsx` | `backend/app/api/auth/` | N/A |
| `POST` | `/api/v1/auth/refresh` | `apiClient.js` | `backend/app/api/auth/` | N/A |
| `GET` | `/api/v1/auth/me` | `SettingsPage.jsx` | `backend/app/api/auth/` | N/A |
| `POST` | `/api/v1/datasets/upload` | `UploadPage.jsx` | `backend/app/api/datasets/` | `ml_engine/data/raw/` |
| `GET` | `/api/v1/datasets` | `DatasetManagementPage.jsx` | `backend/app/api/datasets/` | N/A |
| `GET` | `/api/v1/datasets/{dataset_id}` | `DatasetManagementPage.jsx` | `backend/app/api/datasets/` | N/A |
| `DELETE`| `/api/v1/datasets/{dataset_id}` | `DatasetManagementPage.jsx` | `backend/app/api/datasets/` | Storage cleanup |
| `POST` | `/api/v1/datasets/{dataset_id}/validate` | `UploadPage.jsx` | `backend/app/api/datasets/` | `ml_engine/preprocessing/validator.py` |
| `POST` | `/api/v1/datasets/{dataset_id}/preprocess` | `TrainingPage.jsx` | `backend/app/api/datasets/` | `ml_engine/preprocessing/cleaner.py` |
| `POST` | `/api/v1/forecasting/train/retail` | `TrainingPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/pipelines/training_pipeline.py` |
| `POST` | `/api/v1/forecasting/train/electricity` | `TrainingPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/pipelines/training_pipeline.py` |
| `GET` | `/api/v1/forecasting/models` | `ModelsPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/artifacts/metadata/` |
| `GET` | `/api/v1/forecasting/models/{id}/evaluation` | `ModelsPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/evaluation/evaluator.py` |
| `POST` | `/api/v1/forecasting/predict/retail` | `RetailPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/pipelines/inference_pipeline.py` |
| `POST` | `/api/v1/forecasting/predict/electricity` | `ElectricityPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/pipelines/inference_pipeline.py` |
| `POST` | `/api/v1/forecasting/hybrid-forecast` | `ForecastPage.jsx` | `backend/app/api/forecasting/` | `ml_engine/hybrid_engine/hybrid_forecaster.py` |
| `POST` | `/api/v1/explainability/shap` | `XAIPage.jsx` | `backend/app/api/explainability/` | `ml_engine/explainability/shap_explainer.py` |
| `POST` | `/api/v1/explainability/lime` | `XAIPage.jsx` | `backend/app/api/explainability/` | `ml_engine/explainability/lime_explainer.py` |
| `GET` | `/api/v1/dashboard/overview` | `DashboardPage.jsx` | `backend/app/api/dashboard/` | N/A |
| `GET` | `/api/v1/dashboard/retail` | `RetailPage.jsx` | `backend/app/api/dashboard/` | N/A |
| `GET` | `/api/v1/dashboard/electricity` | `ElectricityPage.jsx` | `backend/app/api/dashboard/` | N/A |
| `POST` | `/api/v1/reports/generate` | `ReportsPage.jsx` | `backend/app/api/reports/` | N/A |
| `GET` | `/api/v1/reports/{id}/download` | `ReportsPage.jsx` | `backend/app/api/reports/` | Storage Stream |
| `GET` | `/api/v1/settings` | `SettingsPage.jsx` | `backend/app/api/settings/` | N/A |
| `PUT` | `/api/v1/settings` | `SettingsPage.jsx` | `backend/app/api/settings/` | N/A |
| `GET` | `/api/v1/health` | Monitoring | `backend/app/api/health/` | Health check |
