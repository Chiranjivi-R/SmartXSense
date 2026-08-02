import pandas as pd
import pytest

from ml_engine.splitting import TimeSeriesSplitError, chronological_split


def _frame(count: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=count, freq="D"), "value": range(count)})


def test_chronological_split_orders_partitions_and_keeps_timestamps_together():
    frame = pd.concat([_frame(), _frame().assign(series="second")], ignore_index=True).sample(frac=1, random_state=3)
    split = chronological_split(frame, train_ratio=.6, validation_ratio=.2, test_ratio=.2)
    assert len(split.train["timestamp"].unique()) == 6
    assert len(split.validation["timestamp"].unique()) == 2
    assert len(split.test["timestamp"].unique()) == 2
    assert split.train["timestamp"].max() < split.validation["timestamp"].min() < split.test["timestamp"].min()
    assert not set(split.train.timestamp) & set(split.validation.timestamp)


@pytest.mark.parametrize("ratios", [(0.7, .2, .2), (0, .5, .5), (True, .2, -.2)])
def test_chronological_split_rejects_invalid_ratios(ratios):
    with pytest.raises(TimeSeriesSplitError):
        chronological_split(_frame(), train_ratio=ratios[0], validation_ratio=ratios[1], test_ratio=ratios[2])


def test_chronological_split_rejects_too_few_timestamps():
    with pytest.raises(TimeSeriesSplitError, match="at least 3"):
        chronological_split(_frame(2))
