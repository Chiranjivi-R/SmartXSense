# SmartXSense Machine Learning API Contract

This document specifies the REST API contract for the SmartXSense Machine Learning backend service.

---

## Base Path & Authentication

All forecasting and explainability routes are registered under Flask blueprint prefixes:

- **Forecasting Base Path**: `/api/v1/forecasting`
- **Explainability Base Path**: `/api/v1/explainability`

---

## 1. Forecast Endpoint

Train/predict with a single model or hybrid ensemble and receive evaluation metrics and chart-ready test predictions.

- **HTTP Method**: `POST`
- **URL**: `/api/v1/forecasting/forecast` (also mapped to `/api/v1/forecasting/train`)
- **Content-Type**: `application/json`

### Request Payload Example

```json
{
  "domain": "retail",
  "features_dir": "ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4",
  "model": "random_forest",
  "n_estimators": 200,
  "prediction_limit": 100
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `domain` | String | Yes | — | `"retail"` or `"electricity"` |
| `features_dir` | String | Yes | — | Relative path to Phase 5C feature directory |
| `model` | String | Yes | — | `"naive"`, `"seasonal_naive"`, `"arima"`, `"random_forest"`, `"xgboost"`, `"lstm"`, `"cnn_lstm"`, or `"hybrid"` |
| `prediction_limit` | Integer | No | `100` | Max predictions to return for plotting (range: 1–1000) |
| `n_estimators` | Integer | No | `200` | Tree count for RF / XGBoost |
| `max_depth` | Integer | No | `null` | Max tree depth |
| `learning_rate` | Float | No | `0.05` | Learning rate for XGBoost |
| `components` | Array[String] | No | `null` | Component models for `"hybrid"` |
| `hybrid_weighting` | String | No | `"inverse_rmse"` | Hybrid weighting strategy (`"inverse_rmse"` or `"equal"`) |

---

### Response Example (`200 OK`)

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "model": "random_forest",
    "domain": "retail",
    "target": "sales",
    "configuration": {
      "n_estimators": 200,
      "max_depth": null,
      "random_state": 42
    },
    "feature_count": 42,
    "feature_names": ["lag_1", "rolling_mean_7", "temperature_c"],
    "training_rows": 767,
    "validation_rows": 164,
    "test_rows": 165,
    "validation_metrics": {
      "mae": 8.17,
      "rmse": 11.25,
      "mape": 0.19,
      "r2": 0.84
    },
    "test_metrics": {
      "mae": 14.62,
      "rmse": 24.70,
      "mape": 0.23,
      "r2": 0.72
    },
    "prediction_count": 100,
    "total_test_predictions": 165,
    "predictions": [
      {
        "timestamp": "2024-07-20T00:00:00+00:00",
        "actual": 80.0,
        "predicted": 103.9
      },
      {
        "timestamp": "2024-07-29T00:00:00+00:00",
        "actual": 13.0,
        "predicted": 13.95
      }
    ]
  }
}
```

---

## 2. Compare Endpoint

Compare candidate forecasting models based strictly on **validation metrics** and obtain adaptive recommendations.

- **HTTP Method**: `POST`
- **URL**: `/api/v1/forecasting/compare`
- **Content-Type**: `application/json`

### Request Payload Example

```json
{
  "domain": "electricity",
  "features_dir": "ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25",
  "selection_metric": "rmse",
  "models": ["arima", "random_forest", "xgboost", "hybrid"]
}
```

---

### Response Example (`200 OK`)

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "domain": "electricity",
    "selection_metric": "rmse",
    "selection_split": "validation",
    "recommended_model": "hybrid",
    "ranking": [
      {
        "rank": 1,
        "model": "hybrid",
        "validation_metrics": { "rmse": 3457.96, "mae": 2508.05, "mape": 0.013, "r2": 0.97 },
        "test_metrics": { "rmse": 4566.47, "mae": 3374.92, "mape": 0.017, "r2": 0.96 }
      },
      {
        "rank": 2,
        "model": "random_forest",
        "validation_metrics": { "rmse": 3830.27, "mae": 2698.40, "mape": 0.014, "r2": 0.96 },
        "test_metrics": { "rmse": 5063.16, "mae": 3617.50, "mape": 0.018, "r2": 0.95 }
      }
    ],
    "adaptive_selection": {
      "selected_model": "hybrid",
      "selection_metric": "rmse",
      "selection_split": "validation",
      "reason": "Selected model 'hybrid' with optimal validation RMSE of 3457.9631 among 4 candidate models.",
      "validation_metrics": { "rmse": 3457.96, "mae": 2508.05, "mape": 0.013, "r2": 0.97 }
    }
  }
}
```

---

## 3. Explain Endpoint

Generate SHAP feature importances or LIME local feature attributions for tree-based forecasters (`random_forest`, `xgboost`).

- **HTTP Method**: `POST`
- **URL**: `/api/v1/explainability/explain`
- **Content-Type**: `application/json`

### Global Explanation Request

```json
{
  "domain": "retail",
  "features_dir": "ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4",
  "model": "random_forest",
  "explanation_type": "global",
  "method": "shap"
}
```

---

## Common Error Responses

### `400 Bad Request`

Returned for path traversal, invalid parameters, missing files, or unsupported model/domain combinations:

```json
{
  "status": "error",
  "code": 400,
  "message": "Parameter 'prediction_limit' must be between 1 and 1000"
}
```

---

## Recommended Frontend Integration Workflow

1. User selects domain (`"retail"` or `"electricity"`).
2. User selects model or chooses **Auto Select**.
3. If Auto Select is chosen, call `POST /api/v1/forecasting/compare` to get `recommended_model`.
4. Call `POST /api/v1/forecasting/forecast` with the selected model.
5. Render summary metric cards using `validation_metrics` and `test_metrics`.
6. Plot the Actual vs Predicted time-series graph using `data.predictions` (`timestamp`, `actual`, `predicted`).
7. For tree models (`random_forest`, `xgboost`), call `POST /api/v1/explainability/explain` to populate the feature importance / XAI panel.
