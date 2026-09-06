# SmartXSense Held-Out Test Model Comparison Audit Report

**Audit Date:** 2026-09-06 19:09:58 UTC
**Auditor:** Senior ML Visualization & Results Engineer
**Scope:** Verification and visualization of all applicable forecasting models on canonical held-out test splits for Retail and Electricity domains.

---

## 1. Data Source Used
- **Retail Domain Dataset:**
  - Directory: `ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4`
  - Target: `sales`
  - Granularity: Daily
  - Test Split File: `test.parquet`
- **Electricity Domain Dataset:**
  - Directory: `ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25`
  - Target: `demand_mw`
  - Granularity: Hourly
  - Test Split File: `test.parquet`

---

## 2. Retail Test Period
- **Start Timestamp:** `2024-07-20 00:00:00`
- **End Timestamp:** `2024-12-31 00:00:00`
- **Span:** 2024-07-20 to 2024-12-31 (165 days)

---

## 3. Electricity Test Period
- **Start Timestamp:** `2024-11-10 10:00:00`
- **End Timestamp:** `2026-03-31 23:00:00`
- **Span:** 2024-11-10 10:00:00 to 2026-03-31 23:00:00 (12,158 hours / ~1.4 years)

---

## 4. Retail Test Observation Count
- **Total Rows:** `165` observations
- **Completeness:** 100% (zero missing or clipped timestamps)

---

## 5. Electricity Test Observation Count
- **Total Rows:** `12,158` observations
- **Completeness:** 100% (zero missing or clipped timestamps)

---

## 6. Models Plotted for Retail
Every applicable model identified by the canonical applicability profile was evaluated and plotted:
| Model Key | Display Name | Test RMSE | Test MAE | Test R² | Prediction Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `naive` | Naive | 60.3281 | 41.5636 | -0.6778 | canonical_pipeline_evaluation |
| `seasonal_naive` | Seasonal Naive | 64.3525 | 44.6788 | -0.9091 | canonical_pipeline_evaluation |
| `random_forest` | Random Forest | 23.9564 | 13.0832 | 0.7354 | pre_trained_artifact |
| `xgboost` | XGBoost | 22.2668 | 12.4570 | 0.7714 | pre_trained_artifact |
| `lstm` | LSTM | 45.5496 | 31.3671 | 0.0435 | pre_trained_artifact |
| `cnn_lstm` | CNN-LSTM | 46.7876 | 31.3229 | -0.0092 | pre_trained_artifact |
| `hybrid` | Hybrid | 22.9983 | 12.6693 | 0.7562 | canonical_pipeline_evaluation |

---

## 7. Models Plotted for Electricity
All 10 models supported by the electricity domain architecture were evaluated and plotted:
| Model Key | Display Name | Test RMSE | Test MAE | Test R² | Prediction Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `naive` | Naive | 6928.7417 | 5239.0464 | 0.9119 | canonical_pipeline_evaluation |
| `seasonal_naive` | Seasonal Naive | 6938.7386 | 4979.3876 | 0.9116 | canonical_pipeline_evaluation |
| `arima` | ARIMA | 4816.9650 | 3689.7813 | 0.9574 | canonical_pipeline_evaluation |
| `sarima` | SARIMA | 1585.3242 | 1112.3089 | 0.9954 | canonical_pipeline_evaluation |
| `random_forest` | Random Forest | 4998.1783 | 3573.1243 | 0.9541 | pre_trained_artifact |
| `xgboost` | XGBoost | 6330.9437 | 4136.3340 | 0.9264 | pre_trained_artifact |
| `trend_xgboost` | Trend XGBoost | 2552.5325 | 1890.4526 | 0.9880 | pre_trained_artifact |
| `lstm` | LSTM | 4861.4063 | 3824.7585 | 0.9566 | pre_trained_artifact |
| `cnn_lstm` | CNN-LSTM | 4712.0519 | 3681.4013 | 0.9592 | canonical_pipeline_evaluation |
| `hybrid` | Hybrid | 4096.2670 | 3094.2462 | 0.9692 | canonical_pipeline_evaluation |

---

## 8. Models Excluded and Why
- **Retail Domain Exclusions:**
  1. **ARIMA (`arima`):** Excluded under reason code `INTERLEAVED_PANEL`. The canonical Retail dataset contains interleaved multi-store/multi-department panel data, which violates the requirement of single continuous stationary/differenced 1D time series.
  2. **SARIMA (`sarima`):** Excluded under reason code `INTERLEAVED_PANEL`. Seasonal ARIMA requires regular, contiguous seasonality over a single series. Interleaving introduces artificial phase shifts.
  3. **Trend XGBoost (`trend_xgboost`):** Excluded under reason code `NO_HOURLY_LAG_STRUCTURE`. Trend XGBoost and piecewise spline knots are specifically engineered for continuous hourly electricity load series (requiring `lag_1`, `lag_24`, `lag_168`).
- **Electricity Domain Exclusions:**
  - None. All 10 models in the SmartXSense ecosystem are applicable to the continuous hourly electricity dataset.

---

## 9. Confirmation that Test Predictions Were Used
- **Confirmed:** All predictions evaluated and plotted were generated strictly on the held-out `test.parquet` partition using `combined_train_val` history where required by autoregressive models.
- Validation split data (`validation.parquet`) was **NOT** plotted as test predictions.

---

## 10. Confirmation that No Future Predictions Were Used
- **Confirmed:** Zero future forecast horizon rows (t > test_end) are present in the visualization or prediction CSV files. Every predicted row corresponds to an actual observation in the ground truth test split.

---

## 11. Timestamp Alignment Result
- **Retail Domain:** 165 timestamps strictly monotonically increasing from `2024-07-20` to `2024-12-31`. Perfect 1:1 row alignment with ground truth actual sales.
- **Electricity Domain:** 12,158 timestamps strictly monotonically increasing from `2024-11-10 10:00:00` to `2026-03-31 23:00:00`. Perfect 1:1 row alignment with ground truth actual electricity demand.

---

## 12. Prediction Length Validation
- **Retail:** Actuals length = `165`, every model prediction length = `165`.
- **Electricity:** Actuals length = `12,158`, every model prediction length = `12,158`.
- No truncated arrays, no padded arrays, no missing values.

---

## 13. Output Files Created
1. `result/model_comparison_heldout_test.png` (High-resolution 300 DPI 1x2 side-by-side comparison figure)
2. `result/model_comparison_heldout_test.svg` (Lossless vector figure for presentations and academic papers)
3. `result/model_comparison_heldout_test.pdf` (Print-ready publication PDF)
4. `result/retail_model_comparison_test_predictions.csv` (Aligned timestamps, actual sales, and 7 model predictions)
5. `result/electricity_model_comparison_test_predictions.csv` (Aligned timestamps, actual demand, and 10 model predictions)
6. `result/model_comparison_metadata.json` (Machine-readable metadata, evaluation metrics, and validation logs)
7. `result/model_comparison_audit.md` (This comprehensive audit report)

---

## 14. Limitations
- The Retail test set spans 165 days (daily frequency), representing a single interleaved aggregation series.
- The Electricity test set spans 12,158 hourly points (~1.4 years), showing multi-seasonal daily/weekly cycles and seasonal weather swings. To maintain readability while plotting the complete unclipped series, minor line transparency (alpha 0.85) and distinct linestyles are utilized.

---

## Final Status
**READY FOR RESULT FIGURE**
