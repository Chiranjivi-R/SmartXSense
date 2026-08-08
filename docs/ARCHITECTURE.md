# SmartXSense Architecture

## End-to-end pipeline

```
Data ingestion (upload or demo feature artifacts)
        ↓
Schema validation (domain-specific)
        ↓
Preprocessing / cleaning
        ↓
Leakage-safe feature engineering
        ↓
Chronological train / validation / test splits
        ↓
Model training
        ↓
Validation-based selection (Compare / Auto Select)
        ↓
Held-out test evaluation (unbiased metrics + actual vs predicted)
        ↓
Controlled model persistence (compatible artifact reuse)
        ↓
Future forecasting (predicted-only horizon)
        ↓
Explainability (SHAP global / LIME local for tree models)
        ↓
Forecast history (metadata JSON) + Reports/Dashboard summaries
        ↓
Client-side CSV/JSON export
        ↓
Flask REST APIs → React frontend
```

## Guarantees

| Guarantee | How it is enforced |
|---|---|
| Validation for selection | Compare/Auto Select use validation metrics only |
| Unbiased test | Test metrics computed on held-out test split after selection |
| No future actuals | Future API returns `{timestamp, predicted}` only |
| No preprocess refit on future | Future feature construction uses fitted history policies (calendar + lags; retail LOCF) |
| Artifact safety | Compatibility checks on domain/model/target/features_id/config; corrupt → retrain |
| Upload feature parity | Upload preprocess reuses `prepare_forecasting_data` / feature materialization |
| History size | History never stores prediction arrays |
| Export safety | Browser Blob downloads; path-like keys stripped from JSON |

## Layers

### Data & features (`ml_engine/`)

- Preprocessing schemas and cleaners
- Feature engineering CLI + upload materialization
- Demo feature directories under `ml_engine/data/features/` (local; gitignored when generated)

### Models (`ml_engine/models/`)

- Baselines, ARIMA, RF, XGBoost, LSTM, CNN-LSTM, Hybrid
- `artifact_store.py` — persistence
- `future_forecasting.py` — out-of-sample horizons
- `capability_matrix.py` — authoritative capability flags
- `trainer.py` — training + evaluation orchestration

### Evaluation & explainability

- Metrics + AdaptiveModelSelector
- SHAP / LIME wrappers for RF and XGBoost

### History

- `ml_engine/history/` — file-based run index (academic-project simplicity; not DB ForecastJob tables)

### Backend (`backend/app/`)

- Auth + datasets (SQLAlchemy)
- Forecasting / future / compare / history
- Explainability
- Dashboard / reports / settings / health (file + capability derived)

### Frontend (`frontend/src/`)

- Shell navigation in `App.jsx`
- `ForecastDashboard` — core ML UX
- Overview, Explainability, Models, Reports, Settings pages
- `mlApi.js` — centralized API client
- `forecastExport.js` — client-side export helpers

## Controlled storage roots

| Path | Purpose |
|---|---|
| `ml_engine/artifacts/models/` | Persisted models |
| `ml_engine/artifacts/history/` | Forecast history JSON |
| `storage/uploads/` | Uploaded datasets |
| `ml_engine/data/processed/` | Cleaned parquet |
| `ml_engine/data/features/` | Feature split directories |

API callers never supply arbitrary absolute artifact paths for load/save.
