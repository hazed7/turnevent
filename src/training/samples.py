from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import numpy.typing as npt
import pandas as pd

from contracts.market import WINDOW_LENGTH, Column
from core.errors import TrainingError
from dataset.cleaning import CleanDataset
from features.builder import FEATURE_NAMES
from labels.generator import LabelColumn, LabelSet

type Codes = npt.NDArray[np.int64]


class SampleColumn(StrEnum):
    EVENT_TIMESTAMP = auto()
    SPAN_START_MS = auto()
    SPAN_END_MS = auto()


@dataclass(frozen=True, slots=True)
class Samples:
    keys: pd.DataFrame
    features: pd.DataFrame
    turn_event: Codes
    turn_state: Codes

    def __len__(self) -> int:
        return len(self.keys)

    def take(self, rows: npt.NDArray[np.bool_]) -> "Samples":
        return Samples(
            keys=self.keys.loc[rows].reset_index(drop=True),
            features=self.features.loc[rows].reset_index(drop=True),
            turn_event=self.turn_event[rows],
            turn_state=self.turn_state[rows],
        )


def assemble_samples(
    clean: CleanDataset,
    features: pd.DataFrame,
    labels: LabelSet,
    *,
    prediction_lag_bars: int = 0,
) -> Samples:
    if not 0 <= prediction_lag_bars <= labels.config.future_bars:
        raise TrainingError(f"prediction lag must be 0..{labels.config.future_bars} bars")
    labelled = labels.labelled()[
        [
            Column.SYMBOL,
            Column.TIMESTAMP,
            LabelColumn.TURN_EVENT,
            LabelColumn.TURN_STATE,
            LabelColumn.LABEL_AVAILABLE_TIMESTAMP,
        ]
    ]
    merged = (
        labelled.merge(
            _timing(clean, labels, prediction_lag_bars),
            on=[Column.SYMBOL, Column.TIMESTAMP],
            how="inner",
            validate="one_to_one",
        )
        .rename(columns={Column.TIMESTAMP: SampleColumn.EVENT_TIMESTAMP})
        .rename(columns={_PREDICTED_AT: Column.TIMESTAMP})
        .merge(features, on=[Column.SYMBOL, Column.TIMESTAMP], how="inner", validate="one_to_one")
        .sort_values([Column.TIMESTAMP, Column.SYMBOL], kind="stable", ignore_index=True)
    )
    if merged.empty:
        raise TrainingError("no labelled row has a full feature window")
    return Samples(
        keys=pd.DataFrame(
            {
                Column.SYMBOL: merged[Column.SYMBOL],
                Column.TIMESTAMP: merged[Column.TIMESTAMP],
                SampleColumn.EVENT_TIMESTAMP: merged[SampleColumn.EVENT_TIMESTAMP],
                SampleColumn.SPAN_START_MS: merged[SampleColumn.SPAN_START_MS],
                SampleColumn.SPAN_END_MS: merged[LabelColumn.LABEL_AVAILABLE_TIMESTAMP].astype(
                    np.int64
                ),
            },
        ),
        features=merged[list(FEATURE_NAMES)],
        turn_event=merged[LabelColumn.TURN_EVENT].to_numpy(dtype=np.int64),
        turn_state=merged[LabelColumn.TURN_STATE].to_numpy(dtype=np.int64),
    )


_PREDICTED_AT = "predicted_at"


def _timing(clean: CleanDataset, labels: LabelSet, lag: int) -> pd.DataFrame:
    bars = clean.frame[[Column.SYMBOL, Column.TIMESTAMP]]
    timestamps = bars.groupby(Column.SYMBOL)[Column.TIMESTAMP]
    lookback = max(WINDOW_LENGTH - 1 - lag, labels.config.history_bars)
    return (
        bars.assign(
            **{
                _PREDICTED_AT: timestamps.shift(-lag),
                SampleColumn.SPAN_START_MS: timestamps.shift(lookback),
            },
        )
        .dropna()
        .astype({_PREDICTED_AT: np.int64, SampleColumn.SPAN_START_MS: np.int64})
    )
