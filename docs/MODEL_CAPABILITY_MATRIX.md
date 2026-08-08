# Model capability matrix

Authoritative source: `ml_engine/models/capability_matrix.py`  
Exposed via: `GET /api/v1/dashboard/models`

| Model | Retail | Electricity | Test Forecast | Future | Persistence | Explainability | Compare/Auto Select |
|---|---|---|---|---|---|---|---|
| Naive | yes | yes | yes | no | yes | no | optional |
| Seasonal Naive | yes | yes | yes | no | yes | no | optional |
| ARIMA | no | yes | yes | electricity | yes | no | yes (elec default) |
| Random Forest | yes | yes | yes | retail, electricity | yes | yes | yes |
| XGBoost | yes | yes | yes | retail, electricity | yes | yes | yes |
| LSTM | yes | yes | yes | no | yes | no | optional |
| CNN-LSTM | yes | yes | yes | no | yes | no | optional |
| Hybrid | yes | yes | yes | no | no | no | yes |

## Notes

- **Test Forecast** = held-out evaluation with actual vs predicted.
- **Future** = out-of-sample horizon; predictions never include actuals.
- **Persistence** = safe artifact save/load under controlled roots. Hybrid is excluded because fitted components are not retained.
- **Explainability** = SHAP/LIME for Random Forest and XGBoost only.
- **Compare/Auto Select** defaults exclude expensive deep models; they can still be requested explicitly where supported by the trainer.
- Accuracy/MAPE values are **never** fabricated in the catalog — run a forecast to evaluate.
