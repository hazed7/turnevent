from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final

import numpy as np
import pandas as pd

from contracts.market import BAR_INTERVAL_MS, FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, Column
from core.errors import DataContractError
from core.frozen import FrozenModel

CLEANING_POLICY_VERSION: Final = "1.0.0"

_TIMESTAMP_PATTERN: Final = r"\d{1,18}"
_PRICES: Final = [Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE]
_KEY: Final = [Column.SYMBOL, Column.TIMESTAMP]


class DropReason(StrEnum):
    MISSING_VALUE = auto()
    INVALID_VALUE = auto()
    OUT_OF_DOMAIN = auto()
    INCONSISTENT_OHLC = auto()
    DUPLICATE = auto()
    CONFLICTING_DUPLICATE = auto()


class CleaningReport(FrozenModel):
    policy_version: str = CLEANING_POLICY_VERSION
    rows_in: int
    rows_out: int
    dropped: Mapping[DropReason, int]
    missing_values: Mapping[str, int]
    invalid_values: Mapping[str, int]
    non_monotonic_steps: int
    off_grid_timestamps: int


@dataclass(frozen=True, slots=True)
class CleanDataset:
    frame: pd.DataFrame
    report: CleaningReport

    def __post_init__(self) -> None:
        require_clean(self.frame)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.frame[Column.SYMBOL].unique())

    @property
    def period_ms(self) -> tuple[int, int]:
        timestamps = self.frame[Column.TIMESTAMP]
        return int(timestamps.min()), int(timestamps.max())


def clean_historical(raw: pd.DataFrame, *, interval_ms: int = BAR_INTERVAL_MS) -> CleanDataset:
    frame, missing, invalid = _parse(raw)
    dropped: dict[DropReason, int] = {}

    def drop(reason: DropReason, rows: pd.Series) -> None:
        nonlocal frame
        dropped[reason] = int(rows.sum())
        frame = frame.loc[~rows]

    drop(DropReason.MISSING_VALUE, missing.any(axis=1))
    drop(DropReason.INVALID_VALUE, invalid.loc[frame.index].any(axis=1))
    drop(DropReason.OUT_OF_DOMAIN, ~_in_domain(frame))
    drop(DropReason.INCONSISTENT_OHLC, ~_ohlc_consistent(frame))

    non_monotonic = int((frame.groupby(Column.SYMBOL)[Column.TIMESTAMP].diff() < 0).sum())

    frame = frame.sort_values(_KEY, kind="stable")
    drop(DropReason.DUPLICATE, frame.duplicated(keep="first"))
    drop(DropReason.CONFLICTING_DUPLICATE, frame.duplicated(_KEY, keep=False))
    frame = frame.reset_index(drop=True)

    if frame.empty:
        raise DataContractError("no usable rows remain after cleaning")

    report = CleaningReport(
        rows_in=len(raw),
        rows_out=len(frame),
        dropped=dropped,
        missing_values=_count_per_column(missing),
        invalid_values=_count_per_column(invalid),
        non_monotonic_steps=non_monotonic,
        off_grid_timestamps=int((frame[Column.TIMESTAMP] % interval_ms != 0).sum()),
    )
    return CleanDataset(frame=frame, report=report)


def require_clean(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != MODEL_INPUT_COLUMNS:
        raise DataContractError(
            f"expected columns {MODEL_INPUT_COLUMNS}, got {tuple(frame.columns)}"
        )

    if frame.empty:
        raise DataContractError("clean dataset is empty")
    if frame[Column.TIMESTAMP].dtype != np.int64:
        raise DataContractError("timestamp must be int64")
    if not pd.api.types.is_string_dtype(frame[Column.SYMBOL]):
        raise DataContractError("symbol must be a string column")

    features = frame[list(FEATURE_COLUMNS)]
    if any(dtype != np.float64 for dtype in features.dtypes):
        raise DataContractError("feature columns must be float64")
    if not np.isfinite(features.to_numpy()).all():
        raise DataContractError("feature columns must be finite")

    keys = pd.MultiIndex.from_frame(frame[_KEY])
    if not (keys.is_monotonic_increasing and keys.is_unique):
        raise DataContractError("rows must be sorted by (symbol, timestamp) with unique keys")


def _parse(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = raw.isna()

    timestamps = raw[Column.TIMESTAMP]
    timestamp_ok = timestamps.str.fullmatch(_TIMESTAMP_PATTERN).fillna(value=False).astype(bool)

    features = raw[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype(np.float64)

    invalid = pd.DataFrame(data=False, index=raw.index, columns=raw.columns)
    invalid[Column.TIMESTAMP] = ~timestamp_ok & ~missing[Column.TIMESTAMP]
    invalid[list(FEATURE_COLUMNS)] = ~np.isfinite(features) & ~missing[list(FEATURE_COLUMNS)]

    frame = pd.DataFrame(
        {
            Column.TIMESTAMP: timestamps.where(timestamp_ok, "0").astype(np.int64),
            Column.SYMBOL: raw[Column.SYMBOL],
            **{column: features[column] for column in FEATURE_COLUMNS},
        },
    )
    return frame, missing, invalid


def _in_domain(frame: pd.DataFrame) -> pd.Series:
    return (frame[_PRICES] > 0).all(axis=1) & (frame[Column.VOLUME] >= 0)


def _ohlc_consistent(frame: pd.DataFrame) -> pd.Series:
    body_top = frame[[Column.OPEN, Column.CLOSE]].max(axis=1)
    body_bottom = frame[[Column.OPEN, Column.CLOSE]].min(axis=1)
    return (frame[Column.HIGH] >= body_top) & (frame[Column.LOW] <= body_bottom)


def _count_per_column(mask: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in mask.sum().items()}
