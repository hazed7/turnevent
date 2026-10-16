from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from contracts.market import MODEL_INPUT_COLUMNS
from dataset.cleaning import CleanDataset, clean_historical

type RawBars = Callable[..., pd.DataFrame]


@pytest.fixture
def raw_bars() -> RawBars:
    def build(
        closes: Sequence[float],
        *,
        minutes: Sequence[int] | None = None,
        symbol: str = "A",
        band: float = 0.5,
    ) -> pd.DataFrame:
        minutes = minutes if minutes is not None else range(1, len(closes) + 1)
        rows = [
            (minute * 60_000, symbol, 0.0, close, close + band, close - band, close, 100.0)
            for minute, close in zip(minutes, closes, strict=True)
        ]
        return pd.DataFrame(rows, columns=list(MODEL_INPUT_COLUMNS)).astype("str")

    return build


@pytest.fixture
def make_clean(raw_bars: RawBars) -> Callable[..., CleanDataset]:
    def make(closes: Sequence[float], **kwargs: object) -> CleanDataset:
        return clean_historical(raw_bars(closes, **kwargs))

    return make
