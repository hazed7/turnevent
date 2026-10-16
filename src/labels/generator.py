from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pydantic import PositiveFloat, PositiveInt

from contracts.market import BAR_INTERVAL_MS, Column
from contracts.targets import TurnEvent, TurnState
from core.frozen import FrozenModel
from dataset.cleaning import CleanDataset

LABEL_SPEC_VERSION: Final = "1.0.0"
LABEL_GENERATOR_VERSION: Final = "1.0.0"

type Floats = npt.NDArray[np.float64]
type Bools = npt.NDArray[np.bool_]

_STATE_AFTER_TURN: Final = {
    TurnEvent.TURN_UP: TurnState.BUY_HOLD,
    TurnEvent.TURN_DOWN: TurnState.SELL_HOLD,
}


class LabelConfig(FrozenModel):
    left_bars: PositiveInt = 3
    right_bars: PositiveInt = 3
    min_separation_bars: PositiveInt = 4
    threshold_coefficient: PositiveFloat = 0.65
    scale_window_bars: PositiveInt = 14

    @property
    def history_bars(self) -> int:
        return max(self.left_bars, self.scale_window_bars) + self.min_separation_bars - 1

    @property
    def future_bars(self) -> int:
        return self.right_bars + self.min_separation_bars - 1


class LabelStatus(StrEnum):
    LABELLED = auto()
    INSUFFICIENT_HISTORY = auto()
    NO_CONFIRMATION = auto()
    GAP = auto()
    DISPUTED = auto()


class LabelColumn(StrEnum):
    STATUS = auto()
    TURN_EVENT = auto()
    TURN_STATE = auto()
    EVENT_TIMESTAMP = auto()
    LABEL_AVAILABLE_TIMESTAMP = auto()
    DETECTION_DELAY_MS = auto()


@dataclass(frozen=True, slots=True)
class LabelSet:
    frame: pd.DataFrame
    config: LabelConfig

    def labelled(self) -> pd.DataFrame:
        return self.frame.loc[self.frame[LabelColumn.STATUS] == LabelStatus.LABELLED]


def generate_labels(
    clean: CleanDataset,
    config: LabelConfig,
    *,
    interval_ms: int = BAR_INTERVAL_MS,
) -> LabelSet:
    symbols = clean.frame.groupby(Column.SYMBOL, sort=False)
    frame = pd.concat(
        [_label_symbol(bars, config, interval_ms) for _, bars in symbols], ignore_index=True
    )
    frame[LabelColumn.TURN_STATE] = _turn_state(frame)

    return LabelSet(frame=frame, config=config)


def _label_symbol(bars: pd.DataFrame, config: LabelConfig, interval_ms: int) -> pd.DataFrame:
    timestamps = bars[Column.TIMESTAMP].to_numpy()
    strength, direction, tied = _candidates(bars, config)
    accepted = _locally_strongest(strength, config.min_separation_bars)
    status = _status(timestamps, tied, config, interval_ms)
    labelled = status == LabelStatus.LABELLED

    last = len(bars) - 1
    available = timestamps[np.minimum(np.arange(len(bars)) + config.future_bars, last)]

    def label(values: npt.NDArray[np.int64], dtype: Literal["Int8", "Int64"]) -> pd.Series:
        return pd.Series(values).astype(dtype).where(labelled)

    return pd.DataFrame(
        {
            Column.SYMBOL: bars[Column.SYMBOL].to_numpy(),
            Column.TIMESTAMP: timestamps,
            LabelColumn.STATUS: status,
            LabelColumn.TURN_EVENT: label(np.where(accepted, direction, TurnEvent.NO_TURN), "Int8"),
            LabelColumn.EVENT_TIMESTAMP: label(timestamps, "Int64"),
            LabelColumn.LABEL_AVAILABLE_TIMESTAMP: label(available, "Int64"),
            LabelColumn.DETECTION_DELAY_MS: label(available - timestamps, "Int64"),
        },
    )


def _candidates(bars: pd.DataFrame, config: LabelConfig) -> tuple[Floats, Floats, Bools]:
    left, centre, right = _window(bars[Column.CLOSE].to_numpy(), config)
    others = np.concatenate([left, right], axis=1)
    is_low = (centre[:, None] < others).all(axis=1)
    is_high = (centre[:, None] > others).all(axis=1)
    tied = (centre == others.min(axis=1)) | (centre == others.max(axis=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        rise = np.fmin(left.max(axis=1), right.max(axis=1)) - centre
        fall = centre - np.fmax(left.min(axis=1), right.min(axis=1))
        swing = np.where(is_low, rise, np.where(is_high, fall, np.nan))
        strength = swing / _average_true_range(bars, config.scale_window_bars)

    significant = strength >= config.threshold_coefficient
    direction = np.where(is_low, TurnEvent.TURN_UP, TurnEvent.TURN_DOWN).astype(np.float64)
    return np.where(significant, strength, -np.inf), direction, tied


def _window(values: Floats, config: LabelConfig) -> tuple[Floats, Floats, Floats]:
    left, right = config.left_bars, config.right_bars
    padded = np.pad(values, (left, right), constant_values=np.nan)
    window = sliding_window_view(padded, left + right + 1)
    return window[:, :left], window[:, left], window[:, left + 1 :]


def _average_true_range(bars: pd.DataFrame, window: int) -> Floats:
    high, low, close = (bars[c].to_numpy() for c in (Column.HIGH, Column.LOW, Column.CLOSE))
    previous_close = np.concatenate([close[:1], close[:-1]])
    true_range = np.maximum(high, previous_close) - np.minimum(low, previous_close)
    return pd.Series(true_range).rolling(window).mean().to_numpy()


def _locally_strongest(strength: Floats, separation: int) -> Bools:
    reach = separation - 1
    padded = np.pad(strength, reach, constant_values=-np.inf)
    neighbours = sliding_window_view(padded, 2 * reach + 1)
    before = neighbours[:, :reach].max(axis=1, initial=-np.inf)
    after = neighbours[:, reach + 1 :].max(axis=1, initial=-np.inf)
    return np.isfinite(strength) & (strength > before) & (strength >= after)


def _status(
    timestamps: npt.NDArray[np.int64], tied: Bools, config: LabelConfig, interval_ms: int
) -> npt.NDArray[np.object_]:
    index = np.arange(len(timestamps))
    last = len(timestamps) - 1
    segment = np.concatenate([[0], np.cumsum(np.diff(timestamps) != interval_ms)])
    span_start = np.clip(index - config.min_separation_bars + 1 - config.left_bars, 0, last)
    span_end = np.clip(index + config.future_bars, 0, last)

    status = np.full(len(timestamps), LabelStatus.LABELLED, dtype=object)
    status[tied] = LabelStatus.DISPUTED
    status[segment[span_start] != segment[span_end]] = LabelStatus.GAP
    status[index + config.future_bars > last] = LabelStatus.NO_CONFIRMATION
    status[index < config.history_bars] = LabelStatus.INSUFFICIENT_HISTORY
    return status


def _turn_state(frame: pd.DataFrame) -> pd.Series:
    labelled = frame[LabelColumn.STATUS] == LabelStatus.LABELLED
    run = (~labelled).cumsum()
    turns = frame[LabelColumn.TURN_EVENT].map(_STATE_AFTER_TURN)
    latest = turns.groupby([frame[Column.SYMBOL], run]).ffill()
    return latest.fillna(TurnState.NEUTRAL).astype("Int8").where(labelled)
