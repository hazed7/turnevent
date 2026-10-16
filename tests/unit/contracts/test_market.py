from collections.abc import Callable
from typing import Any

from contracts.market import FEATURE_COLUMNS, HISTORICAL_COLUMNS, MODEL_INPUT_COLUMNS, Column


def test_feature_columns_match_ds_api_header(load_fixture: Callable[[str], Any]) -> None:
    assert list(FEATURE_COLUMNS) == load_fixture("online-ready.json")["featureColumns"]


def test_signal_barrier_is_historical_only() -> None:
    assert Column.SIGNAL_BARRIER in HISTORICAL_COLUMNS
    assert Column.SIGNAL_BARRIER not in MODEL_INPUT_COLUMNS


def test_historical_columns_match_sow_order() -> None:
    assert ", ".join(HISTORICAL_COLUMNS) == (
        "timestamp, symbol, rd_value, open, high, low, close, volume, signal_barrier"
    )
