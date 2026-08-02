"""Deterministic, timestamp-safe train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class TimeSeriesSplitError(ValueError):
    """Raised when a frame cannot be safely divided chronologically."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Non-overlapping chronological partitions and their timestamp column."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    timestamp_column: str = "timestamp"


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0 for value in ratios):
        raise TimeSeriesSplitError("Split ratios must be positive numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise TimeSeriesSplitError("Split ratios must sum to 1.0")


def chronological_split(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_timestamps_per_split: int = 1,
) -> ChronologicalSplit:
    """Split by unique timestamps, never randomly and never through a timestamp.

    Splitting by unique timestamp (rather than row) means panel data cannot place
    observations from the same instant in both training and evaluation partitions.
    """
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    if not isinstance(min_timestamps_per_split, int) or isinstance(min_timestamps_per_split, bool) or min_timestamps_per_split < 1:
        raise TimeSeriesSplitError("min_timestamps_per_split must be a positive integer")
    if timestamp_column not in frame.columns:
        raise TimeSeriesSplitError(f"Missing timestamp column: {timestamp_column}")

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise TimeSeriesSplitError("Timestamp column contains invalid values")
    result[timestamp_column] = timestamps
    result = result.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    unique_timestamps = pd.Index(result[timestamp_column].drop_duplicates())
    minimum = 3 * min_timestamps_per_split
    if len(unique_timestamps) < minimum:
        raise TimeSeriesSplitError(
            f"Dataset has {len(unique_timestamps)} unique timestamps; at least {minimum} are required for splitting"
        )

    train_end = int(len(unique_timestamps) * train_ratio)
    validation_end = train_end + int(len(unique_timestamps) * validation_ratio)
    if train_end < min_timestamps_per_split or validation_end - train_end < min_timestamps_per_split or len(unique_timestamps) - validation_end < min_timestamps_per_split:
        raise TimeSeriesSplitError("Dataset is too small to satisfy the requested split ratios")

    train_times = unique_timestamps[:train_end]
    validation_times = unique_timestamps[train_end:validation_end]
    test_times = unique_timestamps[validation_end:]
    train = result[result[timestamp_column].isin(train_times)].reset_index(drop=True)
    validation = result[result[timestamp_column].isin(validation_times)].reset_index(drop=True)
    test = result[result[timestamp_column].isin(test_times)].reset_index(drop=True)
    # Defensive invariant checks make misuse visible if this implementation changes.
    if set(train[timestamp_column]) & set(validation[timestamp_column]) or set(train[timestamp_column]) & set(test[timestamp_column]) or set(validation[timestamp_column]) & set(test[timestamp_column]):
        raise TimeSeriesSplitError("Split timestamp overlap detected")
    return ChronologicalSplit(train=train, validation=validation, test=test, timestamp_column=timestamp_column)
