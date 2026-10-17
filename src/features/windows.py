from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH, Column
from contracts.online import ReadyWindow
from core.errors import InvalidInputError
from dataset.cleaning import CleanDataset

type Bools = npt.NDArray[np.bool_]
type Ints = npt.NDArray[np.int64]
type Floats = npt.NDArray[np.float64]

_PRICES = [
    FEATURE_COLUMNS.index(column) for column in (Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE)
]
_VOLUME = FEATURE_COLUMNS.index(Column.VOLUME)


@dataclass(frozen=True, slots=True)
class WindowBatch:
    matrices: Floats
    start_ms: Ints
    end_ms: Ints

    def __post_init__(self) -> None:
        n = len(self.matrices)
        if self.matrices.shape[1:] != (WINDOW_LENGTH, len(FEATURE_COLUMNS)):
            raise InvalidInputError(f"windows must be {WINDOW_LENGTH}x{len(FEATURE_COLUMNS)}")
        if self.start_ms.shape != (n,) or self.end_ms.shape != (n,):
            raise InvalidInputError("one start and end timestamp per window")

    def __len__(self) -> int:
        return len(self.matrices)

    def __getitem__(self, rows: slice | Bools) -> "WindowBatch":
        return WindowBatch(self.matrices[rows], self.start_ms[rows], self.end_ms[rows])

    def usable(self) -> Bools:
        finite = np.isfinite(self.matrices).all(axis=(1, 2))
        positive = (self.matrices[:, :, _PRICES] > 0).all(axis=(1, 2))
        volume = (self.matrices[:, :, _VOLUME] >= 0).all(axis=1)
        return finite & positive & volume


def windows_from_ready(items: Sequence[ReadyWindow]) -> WindowBatch:
    return WindowBatch(
        matrices=np.array([item.features for item in items], dtype=np.float64).reshape(
            len(items), WINDOW_LENGTH, len(FEATURE_COLUMNS)
        ),
        start_ms=np.array([item.window_start_timestamp for item in items], dtype=np.int64),
        end_ms=np.array([item.window_end_timestamp for item in items], dtype=np.int64),
    )


def iter_clean_windows(clean: CleanDataset) -> Iterator[tuple[pd.DataFrame, WindowBatch]]:
    for symbol, bars in clean.frame.groupby(Column.SYMBOL, sort=False):
        if len(bars) < WINDOW_LENGTH:
            continue

        timestamps = bars[Column.TIMESTAMP].to_numpy()
        values = bars[list(FEATURE_COLUMNS)].to_numpy()
        windows = sliding_window_view(values, WINDOW_LENGTH, axis=0).transpose(0, 2, 1)
        end = timestamps[WINDOW_LENGTH - 1 :]
        keys = pd.DataFrame({Column.SYMBOL: symbol, Column.TIMESTAMP: end})
        yield keys, WindowBatch(windows, timestamps[: len(end)], end)
