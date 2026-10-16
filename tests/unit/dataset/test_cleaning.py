import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from contracts.market import FEATURE_COLUMNS, HISTORICAL_COLUMNS, Column
from core.errors import DataContractError
from dataset.cleaning import CleanDataset, DropReason, clean_historical, require_clean
from dataset.loader import load_historical

type WriteCsv = Callable[..., Path]

ROW = "60000,A,0.5,10,11,9,10.5,100,1"
NEXT_ROW = "120000,A,0.5,10,11,9,10.5,100,1"


def _clean(write_csv: WriteCsv, *rows: str) -> CleanDataset:
    return clean_historical(load_historical(write_csv(*rows)).frame)


@pytest.mark.parametrize(
    ("bad_row", "reason"),
    [
        ("120000,A,,10,11,9,10.5,100,1", DropReason.MISSING_VALUE),
        ("120000,,0.5,10,11,9,10.5,100,1", DropReason.MISSING_VALUE),
        ("120000,A,0.5,abc,11,9,10.5,100,1", DropReason.INVALID_VALUE),
        ("120000,A,NaN,10,11,9,10.5,100,1", DropReason.INVALID_VALUE),
        ("120000,A,inf,10,11,9,10.5,100,1", DropReason.INVALID_VALUE),
        ("1.2e5,A,0.5,10,11,9,10.5,100,1", DropReason.INVALID_VALUE),
        ("-120000,A,0.5,10,11,9,10.5,100,1", DropReason.INVALID_VALUE),
        ("120000,A,0.5,0,11,9,10.5,100,1", DropReason.OUT_OF_DOMAIN),
        ("120000,A,0.5,10,11,9,10.5,-1,1", DropReason.OUT_OF_DOMAIN),
        ("120000,A,0.5,10,9,11,10.5,100,1", DropReason.INCONSISTENT_OHLC),
        ("120000,A,0.5,10,10.2,9,10.5,100,1", DropReason.INCONSISTENT_OHLC),
        (ROW, DropReason.DUPLICATE),
    ],
)
def test_each_defect_is_dropped_for_its_reason(
    write_csv: WriteCsv, bad_row: str, reason: DropReason
) -> None:
    clean = _clean(write_csv, ROW, bad_row)
    assert len(clean.frame) == 1
    assert {r: n for r, n in clean.report.dropped.items() if n} == {reason: 1}


def test_conflicting_duplicates_are_all_dropped(write_csv: WriteCsv) -> None:
    clean = _clean(write_csv, ROW, NEXT_ROW, "120000,A,0.5,10,11,9,10.5,999,1")
    assert clean.frame[Column.TIMESTAMP].tolist() == [60000]
    assert clean.report.dropped[DropReason.CONFLICTING_DUPLICATE] == 2


def test_timestamps_keep_full_int64_precision(write_csv: WriteCsv) -> None:
    clean = _clean(write_csv, "9007199254740993,A,0.5,10,11,9,10.5,100,1")
    assert clean.frame[Column.TIMESTAMP].tolist() == [9_007_199_254_740_993]


def test_output_is_sorted_typed_and_counts_disorder(write_csv: WriteCsv) -> None:
    clean = _clean(write_csv, "180000,B,0.5,10,11,9,10.5,100,1", NEXT_ROW, ROW)
    keys = clean.frame[[Column.SYMBOL, Column.TIMESTAMP]].to_numpy().tolist()
    assert keys == [["A", 60000], ["A", 120000], ["B", 180000]]
    assert clean.report.non_monotonic_steps == 1
    assert set(clean.frame[list(FEATURE_COLUMNS)].dtypes) == {np.dtype(np.float64)}
    assert clean.symbols == ("A", "B")
    assert clean.period_ms == (60000, 180000)


def test_off_grid_timestamps_are_reported_not_dropped(write_csv: WriteCsv) -> None:
    clean = _clean(write_csv, ROW, "120001,A,0.5,10,11,9,10.5,100,1")
    assert len(clean.frame) == 2
    assert clean.report.off_grid_timestamps == 1


def test_nothing_usable_raises(write_csv: WriteCsv) -> None:
    with pytest.raises(DataContractError, match="no usable rows"):
        _clean(write_csv, "x,A,0.5,10,11,9,10.5,100,1")


@settings(max_examples=25, deadline=None)
@given(st.lists(st.sampled_from(["1", "-1", "0", "", "junk"]), min_size=3, max_size=3))
def test_signal_barrier_never_affects_the_clean_frame(barriers: list[str]) -> None:
    header = ",".join(HISTORICAL_COLUMNS)
    rows = [f"{60000 * (i + 1)},A,0.5,10,11,9,10.5,100" for i in range(len(barriers))]
    with tempfile.TemporaryDirectory() as directory:
        baseline, varied = Path(directory) / "a.csv", Path(directory) / "b.csv"
        baseline.write_text("\n".join([header, *[f"{row},1" for row in rows]]), encoding="utf-8")
        varied.write_text(
            "\n".join([header, *[f"{row},{b}" for row, b in zip(rows, barriers, strict=True)]]),
            encoding="utf-8",
        )
        expected = clean_historical(load_historical(baseline).frame)
        actual = clean_historical(load_historical(varied).frame)
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.report == expected.report


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns=Column.VOLUME), "expected columns"),
        (lambda f: f.iloc[0:0], "empty"),
        (lambda f: f.astype({Column.TIMESTAMP: np.float64}), "timestamp must be int64"),
        (lambda f: f.assign(symbol=1), "symbol must be"),
        (lambda f: f.astype({Column.VOLUME: np.float32}), "float64"),
        (lambda f: f.assign(volume=np.inf), "finite"),
        (lambda f: f.iloc[::-1], "sorted"),
        (lambda f: pd.concat([f, f]), "sorted"),
    ],
)
def test_require_clean_rejects_broken_frames(
    write_csv: WriteCsv, mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    frame = _clean(write_csv, ROW, NEXT_ROW).frame
    with pytest.raises(DataContractError, match=message):
        require_clean(mutate(frame))
