# Database status (honesty)

SmartXSense uses SQLAlchemy + SQLite (configurable) for **authentication and dataset management**.

Forecast execution, persistence, history, dashboard summaries, and reports for ML runs use **file-based** stores under `ml_engine/artifacts/`.

## Actively used models

| Table / model | Status |
|---|---|
| `users` (`User`) | Active — registration/login/JWT |
| `user_sessions` (`UserSession`) | Active — auth sessions |
| `datasets` (`Dataset`) | Active — upload metadata |
| `dataset_validation` (`DatasetValidation`) | Active — validation reports |
| `dataset_preprocessing` (`DatasetPreprocessing`) | Active — preprocess lineage |

## Defined but unused for the current ML workflow

These SQLAlchemy models remain in the schema for planned/legacy persistence. **Do not activate them** for the current product path. They are **not** used by forecasting, persistence, history, dashboard, or reports:

| Table / model | Status |
|---|---|
| `trained_models` (`TrainedModel`) | Schema-only; models live under `ml_engine/artifacts/models/` |
| `model_evaluation` (`ModelEvaluation`) | Schema-only; metrics returned online from training/evaluation |
| `forecast_jobs` (`ForecastJob`) | Schema-only; history is JSON under `ml_engine/artifacts/history/` |
| `forecast_results` (`ForecastResult`) | Schema-only; predictions are API responses only (not DB rows) |
| `explainability_reports` (`ExplainabilityReport`) | Schema-only; SHAP/LIME via online explain API |
| `generated_reports` (`GeneratedReport`) | Schema-only; reports API aggregates history JSON |
| `system_settings` (`SystemSetting`) | Schema-only; settings API is read-only runtime facts |
| `activity_logs` (`ActivityLog`) | Schema-only; unused |

**Authoritative ML persistence today:** file artifacts + forecast history JSON (not these tables).

## Migrations

No migrations are required for Phases A–H. Do not create migrations solely to activate unused forecast/report tables.

## Health endpoint note

`GET /api/v1/health` reports `database` availability via a lightweight `SELECT 1`.

Forecasting remains usable if the database is unavailable (`overall` may be `degraded` while `ml_engine` stays `available`).
