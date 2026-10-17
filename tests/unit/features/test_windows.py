import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH, Column
from contracts.online import ReadyWindow, parse_envelope
from core.errors import InvalidInputError
from dataset.cleaning import CleanDataset
from features.windows import WindowBatch, iter_clean_windows, windows_from_ready


def _batch(n: int = 2) -> WindowBatch:
    matrices = np.full((n, WINDOW_LENGTH, len(FEATURE_COLUMNS)), 10.0)
    return WindowBatch(matrices, np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))


def test_ready_windows_keep_row_and_column_order(load_fixture: Callable[[str], Any]) -> None:
    payload = load_fixture("online-ready.json")
    items = parse_envelope(json.dumps(payload)).items
    batch = windows_from_ready([item for item in items if isinstance(item, ReadyWindow)])

    assert batch.matrices.shape == (2, WINDOW_LENGTH, len(FEATURE_COLUMNS))
    assert batch.matrices[0].tolist() == payload["items"][0]["features"]
    assert batch.end_ms.tolist() == [item["windowEndTimestamp"] for item in payload["items"]]


def test_clean_windows_are_consecutive_rows_per_symbol(clean: CleanDataset) -> None:
    for keys, batch in iter_clean_windows(clean):
        rows = clean.frame.loc[clean.frame[Column.SYMBOL] == keys[Column.SYMBOL].iloc[0]]
        assert len(batch) == len(rows) - WINDOW_LENGTH + 1
        first = rows.iloc[:WINDOW_LENGTH]
        np.testing.assert_array_equal(batch.matrices[0], first[list(FEATURE_COLUMNS)].to_numpy())
        assert keys[Column.TIMESTAMP].iloc[0] == first[Column.TIMESTAMP].iloc[-1]
        assert batch.start_ms[0] == first[Column.TIMESTAMP].iloc[0]


def test_symbols_shorter_than_a_window_are_skipped(clean: CleanDataset) -> None:
    short = CleanDataset(frame=clean.frame.iloc[: WINDOW_LENGTH - 1], report=clean.report)
    assert list(iter_clean_windows(short)) == []


@pytest.mark.parametrize(
    ("column", "value"),
    [(Column.CLOSE, 0.0), (Column.LOW, -1.0), (Column.VOLUME, -1.0), (Column.RD_VALUE, np.nan)],
)
def test_unusable_windows_are_flagged(column: Column, value: float) -> None:
    batch = _batch()
    batch.matrices[1, 5, FEATURE_COLUMNS.index(column)] = value
    assert batch.usable().tolist() == [True, False]


def test_shape_is_enforced() -> None:
    with pytest.raises(InvalidInputError, match="64x6"):
        WindowBatch(np.zeros((1, 63, 6)), np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
    with pytest.raises(InvalidInputError, match="one start and end"):
        WindowBatch(np.zeros((2, 64, 6)), np.zeros(1, dtype=np.int64), np.zeros(2, dtype=np.int64))
