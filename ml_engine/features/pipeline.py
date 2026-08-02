"""Forecasting-ready feature generation followed by chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_engine.splitting import ChronologicalSplit, chronological_split

from .common import FeatureEngineeringError
from .electricity import build_electricity_features
from .retail import build_retail_features


@dataclass(frozen=True)
class FeaturePreparation:
    """Feature frame, chronological partitions, and data-quality report."""

    features: pd.DataFrame
    split: ChronologicalSplit
    feature_names: list[str]
    report: dict


def prepare_forecasting_data(frame: pd.DataFrame, domain: str, **split_options: object) -> FeaturePreparation:
    """Engineer past-only features over all observations, then split in time."""
    if domain == "retail":
        features, names, report = build_retail_features(frame)
    elif domain == "electricity":
        features, names, report = build_electricity_features(frame)
    else:
        raise FeatureEngineeringError("Unsupported dataset domain")
    split = chronological_split(features, **split_options)
    return FeaturePreparation(features=features, split=split, feature_names=names, report=report)
