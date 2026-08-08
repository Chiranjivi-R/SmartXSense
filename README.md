# SmartXSense

Explainable hybrid demand forecasting for **Retail** and **Electricity** systems.

SmartXSense is a full-stack academic project: leakage-aware feature engineering, chronological evaluation, model persistence, future forecasting, explainability (SHAP/LIME), forecast history, CSV/JSON export, and an API-backed React dashboard.

It is **not** a presentation-only demo. Metrics come from real model evaluation on held-out test splits.

---

## 1. Project overview

SmartXSense supports:

- Dataset upload → validation → preprocessing → feature splits
- Test-evaluation forecasting (actual vs predicted)
- Validation-based model comparison and Auto Select
- Future forecasting (predicted only; no fake actuals)
- Model artifact persistence (train once → load)
- SHAP/LIME explainability for Random Forest and XGBoost
- File-based forecast history + reports/dashboard summaries
- Browser CSV/JSON export of the current forecast result

---

## 2. Architecture

```
Dataset upload / demo feature artifacts
        ↓
Validation + preprocessing
        ↓
Leakage-safe feature engineering + chronological train/val/test
        ↓
Model training (validation metrics for selection)
        ↓
Held-out test evaluation
        ↓
Optional persistence under ml_engine/artifacts/
        ↓
Future forecasting / explainability / history / export
        ↓
Flask APIs → React frontend
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detail.

---

## 3. Technology stack

| Layer | Stack |
|---|---|
| Frontend | React 18, Vite, Recharts |
| Backend | Flask, Flask-JWT-Extended, SQLAlchemy |
| ML | pandas, scikit-learn, XGBoost, statsmodels, PyTorch (LSTM/CNN-LSTM), SHAP/LIME |
| Storage | SQLite (auth/datasets), file artifacts + JSON history |

---

## 4. Repository structure

```
smartxsense/
├── backend/                 Flask API
├── frontend/                React app
├── ml_engine/               Features, models, history, evaluation
├── tests/                   ML + backend integration tests
├── docs/                    Architecture + API contract
├── storage/uploads/         Uploaded datasets (gitignored)
└── README.md
```

---

## 5. Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env` from `.env.example` if present (SECRET_KEY, JWT_SECRET_KEY, DATABASE_URL).

### Frontend

```bash
cd frontend
npm install
```

---

## 6. Backend startup

```bash
cd backend
python run.py
```

API base: `http://127.0.0.1:5000`

Health: `GET /api/v1/health`

---

## 7. Frontend startup

```bash
cd frontend
npm run dev
```

Open: `http://localhost:5173`

Optional: `VITE_API_BASE_URL=http://localhost:5000`

---

## 8. Dataset upload workflow

1. Authenticate (upload flow uses JWT via demo registration/login helpers).
2. `POST /api/v1/datasets/upload`
3. Validate schema for `retail` or `electricity`
4. Preprocess → cleaned parquet + forecasting feature splits when enough history exists
5. Use returned `features_dir` (basename id) for forecast

Short datasets may clean successfully with `features_ready=false`.

---

## 9. Forecast workflow

1. Choose Retail or Electricity (or Forecasts)
2. Select model (Random Forest recommended for demo)
3. Run Forecast → metrics + actual vs predicted
4. Compare Models / Auto Select (validation RMSE)
5. SHAP for RF/XGBoost
6. Optional Future Forecast
7. History + Export CSV/JSON

Demo retail features:

`ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4`

---

## 10. Future forecasting

- Endpoint: `POST /api/v1/forecasting/future`
- Returns `{timestamp, predicted}` only
- Retail RF/XGB supported (exogenous LOCF documented)
- Electricity RF/XGB/ARIMA supported
- LSTM/CNN-LSTM/Hybrid future currently unsupported

---

## 11. Model comparison / Auto Select

- Compare ranks candidates on **validation** metrics only
- Auto Select = compare → forecast with recommended model name
- History stores the **selected** model (never `auto_select`)

---

## 12. Explainable AI

- `POST /api/v1/explainability/explain`
- Global SHAP and local LIME for Random Forest / XGBoost
- Unsupported models return controlled 400 errors

---

## 13. Model persistence

Artifacts under `ml_engine/artifacts/models/` (gitignored):

`{domain}/{model}__{features_id}__{config_fp}/`

Compatible load → `artifact.source=loaded`; else train+save → `trained`. Hybrid persistence unsupported.

---

## 14. Forecast history

File store: `ml_engine/artifacts/history/` (`index.json` + `runs/<run_id>.json`)

Metadata only — **no prediction arrays**.

APIs: `GET /api/v1/forecasting/history`, `GET/DELETE .../history/<run_id>`

---

## 15. Export

Client-side CSV/JSON from the current dashboard result (Blob download). No server export storage.

---

## 16. Testing

```bash
python -m pytest tests/ml_engine -q -p no:cacheprovider
python -m pytest tests/backend -q -p no:cacheprovider
cd frontend && npm run test:export && npm run build
```

---

## 17. Known limitations

- Home tips/notifications are **static guides**, not a live event system
- Future forecasting unsupported for LSTM/CNN-LSTM/Hybrid
- Hybrid does not persist fitted component models
- RF/XGBoost support future forecasting; electricity ARIMA also supports future; SHAP/LIME for RF/XGBoost only (see capability matrix)
- SQLAlchemy auth/dataset tables are active; forecast/report tables (`TrainedModel`, `ForecastJob`, `ForecastResult`, `ModelEvaluation`, `ExplainabilityReport`, `GeneratedReport`, `SystemSetting`, `ActivityLog`) are **schema-only / unused** — ML uses file artifacts + history JSON ([docs/DATABASE_STATUS.md](docs/DATABASE_STATUS.md))
- Deep models are expensive; demos focus on RF/XGB
- Settings are read-only system facts

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ML_API_CONTRACT.md](docs/ML_API_CONTRACT.md)
- [docs/DATABASE_STATUS.md](docs/DATABASE_STATUS.md)
- [docs/MODEL_CAPABILITY_MATRIX.md](docs/MODEL_CAPABILITY_MATRIX.md)
- [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md)

> Note: `docs/api/api_specification.md` is a **superseded legacy draft** and must not be treated as the live contract.

---

*SmartXSense — academic full-stack forecasting system*
