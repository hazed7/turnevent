from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from contracts.market import BAR_INTERVAL_MS, FEATURE_COLUMNS, WINDOW_LENGTH, Column
from core.errors import InvalidInputError
from dataset.cleaning import CleanDataset
from features.windows import Floats, WindowBatch, iter_clean_windows

FEATURE_BUILDER_VERSION: Final = "1.1.0"

_CHUNK_ROWS: Final = 50_000
_SWING_SIDE: Final = 3


class _Bars:
    def __init__(self, batch: WindowBatch) -> None:
        def column(name: Column) -> Floats:
            return batch.matrices[:, :, FEATURE_COLUMNS.index(name)]

        self.rd_value = column(Column.RD_VALUE)
        self.open = column(Column.OPEN)
        self.high = column(Column.HIGH)
        self.low = column(Column.LOW)
        self.close = column(Column.CLOSE)
        self.volume = column(Column.VOLUME)
        self.start_ms = batch.start_ms
        self.end_ms = batch.end_ms

    @property
    def last_close(self) -> Floats:
        return self.close[:, -1]

    @property
    def log_returns(self) -> Floats:
        return np.diff(np.log(self.close), axis=1)


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    description: str
    compute: Callable[[_Bars], Floats]


def _ratio(numerator: Floats, denominator: Floats, default: float) -> Floats:
    safe = np.where(denominator == 0, 1.0, denominator)
    return np.where(denominator == 0, default, numerator / safe)


def _log_return(bars: int) -> Callable[[_Bars], Floats]:
    return lambda b: np.log(b.last_close / b.close[:, -1 - bars])


def _volatility(bars: int) -> Callable[[_Bars], Floats]:
    return lambda b: b.log_returns[:, -bars:].std(axis=1)


def _position_in_range(bars: int) -> Callable[[_Bars], Floats]:
    def compute(b: _Bars) -> Floats:
        low, high = b.low[:, -bars:].min(axis=1), b.high[:, -bars:].max(axis=1)
        return _ratio(b.last_close - low, high - low, default=0.5)

    return compute


def _average_true_range(b: _Bars, bars: int) -> Floats:
    previous_close = b.close[:, -bars - 1 : -1]
    high, low = b.high[:, -bars:], b.low[:, -bars:]
    return (np.maximum(high, previous_close) - np.minimum(low, previous_close)).mean(axis=1)


def _rd_zscore(b: _Bars) -> Floats:
    return _ratio(b.rd_value[:, -1] - b.rd_value.mean(axis=1), b.rd_value.std(axis=1), default=0.0)


def _missing_bars(b: _Bars) -> Floats:
    span_bars = np.floor((b.end_ms - b.start_ms) / BAR_INTERVAL_MS)
    return (span_bars - (WINDOW_LENGTH - 1)).clip(min=0.0)


def _swing(b: _Bars) -> tuple[Floats, Floats, Floats]:
    pivot = -_SWING_SIDE - 1
    return b.close[:, pivot - _SWING_SIDE : pivot], b.close[:, pivot], b.close[:, pivot + 1 :]


def _pivot_low(b: _Bars) -> Floats:
    left, pivot, right = _swing(b)
    return (pivot < np.minimum(left.min(axis=1), right.min(axis=1))).astype(np.float64)


def _pivot_high(b: _Bars) -> Floats:
    left, pivot, right = _swing(b)
    return (pivot > np.maximum(left.max(axis=1), right.max(axis=1))).astype(np.float64)


def _pivot_rise(b: _Bars) -> Floats:
    left, pivot, right = _swing(b)
    rise = np.minimum(left.max(axis=1), right.max(axis=1)) - pivot
    return _ratio(rise, _average_true_range(b, 14), default=0.0)


def _pivot_fall(b: _Bars) -> Floats:
    left, pivot, right = _swing(b)
    fall = pivot - np.maximum(left.min(axis=1), right.min(axis=1))
    return _ratio(fall, _average_true_range(b, 14), default=0.0)


def _mean_volume(b: _Bars, bars: int) -> Floats:
    return np.log1p(b.volume[:, -bars:].mean(axis=1)) - np.log1p(b.volume.mean(axis=1))


FEATURES: Final = (
    Feature("return_1", "log return over the last bar", _log_return(1)),
    Feature("return_3", "log return over the last 3 bars", _log_return(3)),
    Feature("return_5", "log return over the last 5 bars", _log_return(5)),
    Feature("return_10", "log return over the last 10 bars", _log_return(10)),
    Feature("return_20", "log return over the last 20 bars", _log_return(20)),
    Feature("return_63", "log return over the whole window", _log_return(WINDOW_LENGTH - 1)),
    Feature("volatility_5", "std of 1-bar log returns, last 5 bars", _volatility(5)),
    Feature("volatility_20", "std of 1-bar log returns, last 20 bars", _volatility(20)),
    Feature(
        "atr_14",
        "average true range of the last 14 bars / last close",
        lambda b: _average_true_range(b, 14) / b.last_close,
    ),
    Feature(
        "bar_range",
        "(high - low) / close of the last bar",
        lambda b: (b.high[:, -1] - b.low[:, -1]) / b.last_close,
    ),
    Feature(
        "bar_body",
        "(close - open) / close of the last bar",
        lambda b: (b.last_close - b.open[:, -1]) / b.last_close,
    ),
    Feature(
        "upper_wick",
        "(high - max(open, close)) / close of the last bar",
        lambda b: (b.high[:, -1] - np.maximum(b.open[:, -1], b.last_close)) / b.last_close,
    ),
    Feature(
        "lower_wick",
        "(min(open, close) - low) / close of the last bar",
        lambda b: (np.minimum(b.open[:, -1], b.last_close) - b.low[:, -1]) / b.last_close,
    ),
    Feature("range_position_20", "close within the 20-bar low-high range", _position_in_range(20)),
    Feature("range_position_64", "close within the window low-high range", _position_in_range(64)),
    Feature(
        "distance_from_mean_20",
        "log(close / mean close of the last 20 bars)",
        lambda b: np.log(b.last_close / b.close[:, -20:].mean(axis=1)),
    ),
    Feature(
        "rebound_7",
        "log(close / lowest close of the last 7 bars)",
        lambda b: np.log(b.last_close / b.close[:, -7:].min(axis=1)),
    ),
    Feature(
        "pullback_7",
        "log(close / highest close of the last 7 bars)",
        lambda b: np.log(b.last_close / b.close[:, -7:].max(axis=1)),
    ),
    Feature(
        "up_share_5",
        "share of the last 5 bars that closed above the previous close",
        lambda b: (b.log_returns[:, -5:] > 0).mean(axis=1),
    ),
    Feature("rd_value", "rd_value of the last bar", lambda b: b.rd_value[:, -1]),
    Feature("rd_value_lag_1", "rd_value one bar earlier", lambda b: b.rd_value[:, -2]),
    Feature("rd_value_lag_3", "rd_value three bars earlier", lambda b: b.rd_value[:, -4]),
    Feature(
        "rd_value_change_1",
        "rd_value change over the last bar",
        lambda b: b.rd_value[:, -1] - b.rd_value[:, -2],
    ),
    Feature(
        "rd_value_change_5",
        "rd_value change over the last 5 bars",
        lambda b: b.rd_value[:, -1] - b.rd_value[:, -6],
    ),
    Feature(
        "rd_value_mean_5", "mean rd_value, last 5 bars", lambda b: b.rd_value[:, -5:].mean(axis=1)
    ),
    Feature(
        "rd_value_mean_20",
        "mean rd_value, last 20 bars",
        lambda b: b.rd_value[:, -20:].mean(axis=1),
    ),
    Feature(
        "rd_value_std_20",
        "std of rd_value, last 20 bars",
        lambda b: b.rd_value[:, -20:].std(axis=1),
    ),
    Feature("rd_value_zscore_64", "last rd_value as a z-score within the window", _rd_zscore),
    Feature(
        "volume_1",
        "log1p(last volume) - log1p(window mean volume)",
        lambda b: _mean_volume(b, 1),
    ),
    Feature(
        "volume_5",
        "log1p(mean volume of the last 5 bars) - log1p(window mean volume)",
        lambda b: _mean_volume(b, 5),
    ),
    Feature("missing_bars", "1m bars missing between window start and end", _missing_bars),
    Feature(
        "pivot_low",
        "1 if the close 4 bars back is below the 3 closes on each side of it",
        _pivot_low,
    ),
    Feature(
        "pivot_high",
        "1 if the close 4 bars back is above the 3 closes on each side of it",
        _pivot_high,
    ),
    Feature(
        "pivot_rise",
        "smaller swing up from the close 4 bars back, in 14-bar average true ranges",
        _pivot_rise,
    ),
    Feature(
        "pivot_fall",
        "smaller swing down into the close 4 bars back, in 14-bar average true ranges",
        _pivot_fall,
    ),
)
FEATURE_NAMES: Final = tuple(feature.name for feature in FEATURES)


def build_features(batch: WindowBatch) -> pd.DataFrame:
    if not batch.usable().all():
        raise InvalidInputError("windows need finite values, positive prices, volume >= 0")

    bars = _Bars(batch)
    with np.errstate(all="ignore"):
        columns = {feature.name: feature.compute(bars) for feature in FEATURES}

    matrix = pd.DataFrame(columns, columns=list(FEATURE_NAMES))
    if not np.isfinite(matrix.to_numpy()).all():
        raise InvalidInputError("feature builder produced non-finite values")
    return matrix


def build_offline_features(clean: CleanDataset) -> pd.DataFrame:
    parts = [
        pd.concat(
            [
                keys.iloc[start : start + _CHUNK_ROWS].reset_index(drop=True),
                build_features(batch[start : start + _CHUNK_ROWS]),
            ],
            axis=1,
        )
        for keys, batch in iter_clean_windows(clean)
        for start in range(0, len(batch), _CHUNK_ROWS)
    ]
    if not parts:
        raise InvalidInputError(f"no symbol has {WINDOW_LENGTH} clean rows")
    return pd.concat(parts, ignore_index=True)
