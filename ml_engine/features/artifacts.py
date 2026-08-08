"""Persist forecasting feature splits using the existing Phase 5C pipeline."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ml_engine.features.common import FeatureEngineeringError
from ml_engine.features.pipeline import FeaturePreparation, prepare_forecasting_data
from ml_engine.splitting.time_series import TimeSeriesSplitError


def write_feature_preparation(
    prepared: FeaturePreparation,
    output_dir: Path,
) -> Path:
    """Write train/validation/test parquet files and metadata.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared.split.train.to_parquet(output_dir / "train.parquet", index=False)
    prepared.split.validation.to_parquet(output_dir / "validation.parquet", index=False)
    prepared.split.test.to_parquet(output_dir / "test.parquet", index=False)
    metadata = {
        "feature_names": prepared.feature_names,
        "report": prepared.report,
        "split_rows": {
            "train": len(prepared.split.train),
            "validation": len(prepared.split.validation),
            "test": len(prepared.split.test),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return output_dir


def materialize_forecasting_features(
    frame: pd.DataFrame,
    domain: str,
    *,
    output_root: Path | str,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Run existing prepare_forecasting_data and write split artifacts.

    Reuses the same feature engineering used by demo datasets / CLI.
    Does not invent a second preprocessing pipeline.
    """
    domain = domain.lower().strip()
    prepared = prepare_forecasting_data(frame, domain)
    root = Path(output_root)
    features_id = artifact_id or str(uuid.uuid4())
    # Keep directory naming compatible with demo artifacts: features_<uuid>
    if not str(features_id).startswith("features_"):
        features_id = f"features_{features_id}"
    output_dir = root / domain / features_id
    write_feature_preparation(prepared, output_dir)
    return {
        "features_dir": output_dir,
        "features_id": output_dir.name,
        "feature_names": prepared.feature_names,
        "report": prepared.report,
        "train_rows": len(prepared.split.train),
        "validation_rows": len(prepared.split.validation),
        "test_rows": len(prepared.split.test),
    }


__all__ = [
    "FeatureEngineeringError",
    "TimeSeriesSplitError",
    "materialize_forecasting_features",
    "write_feature_preparation",
]
