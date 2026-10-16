from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from contracts.market import HISTORICAL_COLUMNS, Column
from dataset.cleaning import DropReason, clean_historical
from dataset.loader import load_historical
from dataset.synthetic import SyntheticConfig, generate_synthetic, write_synthetic_csv

SMALL = SyntheticConfig(bars_per_symbol=200, gap_rate=0.0, missing_join_rate=0.0)


def test_same_seed_same_data_different_seed_different_data() -> None:
    pd.testing.assert_frame_equal(generate_synthetic(SMALL), generate_synthetic(SMALL))
    other = SMALL.model_copy(update={"seed": SMALL.seed + 1})
    assert not generate_synthetic(SMALL).equals(generate_synthetic(other))


def test_clean_output_follows_the_historical_contract() -> None:
    frame = generate_synthetic(SMALL.model_copy(update={"plant_defects": False}))
    assert tuple(frame.columns) == HISTORICAL_COLUMNS
    assert set(frame[Column.SIGNAL_BARRIER]) == {-1, 1}
    assert (frame[Column.TIMESTAMP] % 60_000 == 0).all()
    assert len(frame) == SMALL.n_symbols * SMALL.bars_per_symbol


def test_planted_defects_are_each_caught_by_cleaning(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    write_synthetic_csv(SMALL, path)

    report = clean_historical(load_historical(path).frame).report

    assert report.dropped == {
        DropReason.MISSING_VALUE: 0,
        DropReason.INVALID_VALUE: 1,
        DropReason.OUT_OF_DOMAIN: 1,
        DropReason.INCONSISTENT_OHLC: 1,
        DropReason.DUPLICATE: 1,
        DropReason.CONFLICTING_DUPLICATE: 2,
    }
    assert report.non_monotonic_steps >= 1


def test_gaps_and_missing_joins_are_generated() -> None:
    config = SyntheticConfig(bars_per_symbol=5_000, n_symbols=1, plant_defects=False)
    frame = generate_synthetic(config)
    assert len(frame) < config.bars_per_symbol
    assert frame[Column.RD_VALUE].isna().any()


def test_start_must_lie_on_the_bar_grid() -> None:
    with pytest.raises(ValidationError, match="1m grid"):
        SyntheticConfig(start_timestamp_ms=1)
