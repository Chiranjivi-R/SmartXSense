# SmartXSense Live Presentation & Demo Runbook

This document provides exact startup instructions, step-by-step presentation flows, technical talking points, and recovery procedures for the SmartXSense demonstration.

---

## A. Pre-Presentation Startup Procedures

### 1. Backend Service (Flask REST API)
Open PowerShell / Terminal 1:
```powershell
cd c:\Users\Chiranjivi R\Desktop\MajorProj\smartxsense\backend
python run.py
```
- **Host**: `0.0.0.0` / `localhost`
- **Port**: `5000`
- **Health Check URL**: `http://localhost:5000/api/v1/health`

### 2. Frontend Application (React + Vite)
Open PowerShell / Terminal 2:
```powershell
cd c:\Users\Chiranjivi R\Desktop\MajorProj\smartxsense\frontend
npm run dev
```
- **Host**: `localhost`
- **Port**: `5173`
- **Application URL**: `http://localhost:5173`

---

## B. Primary Demo Walkthrough Sequence

| Step | Action | Expected Output | Timing |
|---|---|---|---|
| **1** | Open `http://localhost:5173` in browser | SmartXSense Forecast Intelligence Dashboard loads | < 1s |
| **2** | Ensure **🛒 Retail** domain is selected | System pre-loads Retail feature artifacts | Instant |
| **3** | Select **Random Forest** & click **⚡ Run Forecast** | Test metrics appear (MAE ~14.5, RMSE ~24.3, MAPE ~23%, R² ~0.73) & 100-point time series line chart renders | ~1–2s |
| **4** | Click **📊 Compare Models** | Benchmark table populates with ranking across `random_forest`, `xgboost`, and `hybrid` based on validation RMSE | ~5–7s |
| **5** | Select **Auto Select** from model dropdown | System benchmarks candidates, highlights recommended model (**xgboost**), and automatically updates forecast | ~4–5s |
| **6** | Scroll down to **Explainable AI (SHAP)** | SHAP feature importances render horizontal bar chart with top feature drivers | ~8–10s |
| **7** | (Optional) Switch to **⚡ Electricity** domain | Forecast load demand (demand_mw) with metrics (MAPE ~1.8%, R² ~0.95) | ~10–13s |

---

## C. Technical Talking Points for Evaluators

- **Forecast Engine**: *"SmartXSense transforms raw temporal data into tabular feature sets, evaluating multi-horizon demand with low-latency REST inference."*
- **Test Metrics**: *"All reported test metrics (MAE, RMSE, MAPE, R²) are evaluated on held-out test data to guarantee unbiased performance measurement."*
- **Validation-Driven Auto Select**: *"Model selection and recommendation rely strictly on validation split performance, preventing data leakage from the test set."*
- **Hybrid Ensemble**: *"The Hybrid forecaster combines tree ensembles with component models weighted inversely to their validation RMSE."*
- **Explainable AI (SHAP)**: *"Tree SHAP computes global feature attributions, quantifying exactly how lag variables and exogenous features influence model predictions."*

---

## D. Recovery & Backup Procedures

1. **If backend service crashes or port is busy**:
   - Restart backend: `cd backend` -> `python run.py`.
2. **If Auto Select benchmark is slow**:
   - Immediately switch dropdown to **Random Forest** or **XGBoost** to execute single-model forecast directly.
3. **If SHAP calculation takes > 10 seconds**:
   - Focus evaluator attention on the **Actual vs Predicted Series** chart and the **Validation Benchmark** table while SHAP computes.

---

## E. Live Presentation Exclusions (Risky Live Models)

- **Do NOT run PyTorch LSTM or CNN-LSTM models live**: CPU inference for deep neural sequence models can introduce 30+ second delays during live presentation.
- **Recommended Live Models**: `random_forest` (Fastest), `xgboost` (Best Validation Accuracy), `auto_select`, and `hybrid`.
