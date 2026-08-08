"""Manual, non-API entry point for Phase 5C feature artifacts."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import pandas as pd

from .artifacts import materialize_forecasting_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe forecasting feature datasets from processed Parquet data.")
    parser.add_argument("--domain", choices=("retail", "electricity"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Phase 5B processed Parquet file")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_engine/data/features"), help="Artifact root (default: ml_engine/data/features)")
    args = parser.parse_args()
    # Keep CLI behavior identical: prepare then materialize under features_<uuid>
    result = materialize_forecasting_features(
        pd.read_parquet(args.input),
        args.domain,
        output_root=args.output_dir,
        artifact_id=str(uuid.uuid4()),
    )
    print(Path(result["features_dir"]).as_posix())


if __name__ == "__main__":
    main()
