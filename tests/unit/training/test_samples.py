import numpy as np
import pandas as pd
import pytest

from contracts.market import WINDOW_LENGTH, Column
from core.errors import TrainingError
from dataset.cleaning import CleanDataset
from features.builder import FEATURE_NAMES, build_offline_features
from labels.generator import LabelColumn, LabelConfig, generate_labels
from training.samples import SampleColumn, Samples, assemble_samples

LAG = 3


def _bars(clean: CleanDataset, symbol: str) -> pd.DataFrame:
    return clean.frame.loc[clean.frame[Column.SYMBOL] == symbol].reset_index(drop=True)


def test_each_sample_pairs_a_label_with_features_lag_bars_later(
    clean: CleanDataset, samples: Samples
) -> None:
    labelled = generate_labels(clean, LabelConfig()).labelled()
    events = samples.keys[[Column.SYMBOL, SampleColumn.EVENT_TIMESTAMP]].rename(
        columns={SampleColumn.EVENT_TIMESTAMP: Column.TIMESTAMP}
    )
    merged = events.merge(labelled, on=[Column.SYMBOL, Column.TIMESTAMP], how="left")
    np.testing.assert_array_equal(merged[LabelColumn.TURN_EVENT].to_numpy(), samples.turn_event)

    row = samples.keys.iloc[100]
    bars = _bars(clean, row[Column.SYMBOL])
    event = int(bars.index[bars[Column.TIMESTAMP] == row[SampleColumn.EVENT_TIMESTAMP]][0])
    assert row[Column.TIMESTAMP] == bars[Column.TIMESTAMP][event + LAG]

    features = build_offline_features(clean).set_index([Column.SYMBOL, Column.TIMESTAMP])
    expected = features.loc[(row[Column.SYMBOL], row[Column.TIMESTAMP])].to_numpy()
    np.testing.assert_array_equal(samples.features.iloc[100].to_numpy(), expected)


def test_span_covers_feature_window_and_label_horizon(
    clean: CleanDataset, samples: Samples
) -> None:
    row = samples.keys.iloc[100]
    bars = _bars(clean, row[Column.SYMBOL])
    predicted = int(bars.index[bars[Column.TIMESTAMP] == row[Column.TIMESTAMP]][0])
    event = predicted - LAG
    assert (
        row[SampleColumn.SPAN_START_MS] == bars[Column.TIMESTAMP][predicted - (WINDOW_LENGTH - 1)]
    )
    assert row[SampleColumn.SPAN_END_MS] == bars[Column.TIMESTAMP][event + 6]


def test_samples_are_unique_ordered_and_free_of_signal_barrier(samples: Samples) -> None:
    keys = samples.keys[[Column.SYMBOL, Column.TIMESTAMP]]
    assert len(keys.drop_duplicates()) == len(samples)
    assert samples.keys[Column.TIMESTAMP].is_monotonic_increasing
    assert list(samples.features.columns) == list(FEATURE_NAMES)


def test_take_keeps_rows_aligned(samples: Samples) -> None:
    every_other = np.arange(len(samples)) % 2 == 0
    taken = samples.take(every_other)
    pd.testing.assert_frame_equal(taken.keys, samples.keys[every_other].reset_index(drop=True))
    np.testing.assert_array_equal(taken.turn_state, samples.turn_state[every_other])


@pytest.mark.parametrize("lag", [-1, 7])
def test_lag_must_stay_within_the_label_horizon(clean: CleanDataset, lag: int) -> None:
    labels = generate_labels(clean, LabelConfig())
    with pytest.raises(TrainingError, match="prediction lag"):
        assemble_samples(clean, build_offline_features(clean), labels, prediction_lag_bars=lag)


def test_no_overlap_between_features_and_labels_raises(clean: CleanDataset) -> None:
    labels = generate_labels(clean, LabelConfig())
    no_features = pd.DataFrame(columns=[Column.SYMBOL, Column.TIMESTAMP, *FEATURE_NAMES])
    with pytest.raises(TrainingError, match="no labelled row"):
        assemble_samples(clean, no_features, labels)
