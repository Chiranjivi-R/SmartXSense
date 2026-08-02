# Forecasting feature preparation

Phase 5C consumes a Phase 5B processed Parquet dataset, creates target-history features using only earlier observations, then writes chronological `train`, `validation`, and `test` Parquet files.

Run it manually from the project root:

```powershell
python -m ml_engine.features.cli --domain retail --input ml_engine/data/processed/retail/proc_example.parquet
```

Artifacts use a UUID directory under `ml_engine/data/features/<domain>/`; they are generated data and are ignored by Git. The metadata file records feature names, excluded leakage-prone columns, duplicate handling, irregular electricity gaps, and rows with unavailable initial history. No missing periods or target values are manufactured.
