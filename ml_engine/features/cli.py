"""Manual, non-API entry point for Phase 5C feature artifacts."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from .pipeline import prepare_forecasting_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe forecasting feature datasets from processed Parquet data.")
    parser.add_argument("--domain", choices=("retail", "electricity"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Phase 5B processed Parquet file")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_engine/data/features"), help="Artifact root (default: ml_engine/data/features)")
    args = parser.parse_args()
    prepared = prepare_forecasting_data(pd.read_parquet(args.input), args.domain)
    output_dir = args.output_dir / args.domain / f"features_{uuid.uuid4()}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prepared.split.train.to_parquet(output_dir / "train.parquet", index=False)
    prepared.split.validation.to_parquet(output_dir / "validation.parquet", index=False)
    prepared.split.test.to_parquet(output_dir / "test.parquet", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({"feature_names": prepared.feature_names, "report": prepared.report}, indent=2, default=str), encoding="utf-8")
    print(output_dir.as_posix())


if __name__ == "__main__":
    main()
