from pathlib import Path
from typing import Annotated, Final, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from contracts.market import BAR_INTERVAL_MS, FEATURE_COLUMNS, WINDOW_LENGTH, Column
from core.files import write_bytes_atomic
from core.frozen import FrozenModel

type Rate = Annotated[float, Field(ge=0.0, lt=1.0)]

_JAN_1_2024_MS: Final = 1_704_067_200_000
_RD_VALUE_SPAN: Final = 10
_RD_VALUE_NOISE: Final = 0.3
_BARRIER_HORIZON: Final = 5


class SyntheticConfig(FrozenModel):
    seed: int = 7
    n_symbols: PositiveInt = 3
    bars_per_symbol: Annotated[int, Field(ge=WINDOW_LENGTH)] = 20_000
    start_timestamp_ms: NonNegativeInt = _JAN_1_2024_MS
    mean_regime_bars: Annotated[float, Field(gt=1.0)] = 40.0
    drift_scale: PositiveFloat = 4e-4
    volatility: PositiveFloat = 1e-3
    gap_rate: Rate = 0.002
    missing_join_rate: Rate = 0.001
    plant_defects: bool = True

    @model_validator(mode="after")
    def _start_on_bar_grid(self) -> Self:
        if self.start_timestamp_ms % BAR_INTERVAL_MS:
            raise ValueError("start_timestamp_ms must lie on the 1m grid")
        return self


def generate_synthetic(config: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    symbols = [
        _simulate_symbol(rng, config, f"SYM{index:03d}") for index in range(config.n_symbols)
    ]
    frame = pd.concat(symbols, ignore_index=True)

    return _plant_defects(frame) if config.plant_defects else frame


def write_synthetic_csv(config: SyntheticConfig, path: Path) -> None:
    write_bytes_atomic(path, generate_synthetic(config).to_csv(index=False).encode("utf-8"))


def _simulate_symbol(
    rng: np.random.Generator, config: SyntheticConfig, symbol: str
) -> pd.DataFrame:
    n = config.bars_per_symbol
    log_returns = _regime_drift(rng, config) + config.volatility * rng.standard_normal(n)

    close = rng.uniform(20.0, 200.0) * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[close[0] * np.exp(-log_returns[0])], close[:-1]])
    wicks = config.volatility * np.abs(rng.standard_normal((2, n)))
    high = np.maximum(open_, close) * np.exp(wicks[0])
    low = np.minimum(open_, close) * np.exp(-wicks[1])
    volume = rng.lognormal(mean=8.0, sigma=0.5, size=n) * (1.0 + 200.0 * np.abs(log_returns))

    momentum = pd.Series(log_returns).ewm(span=_RD_VALUE_SPAN, adjust=False).mean().to_numpy()
    rd_value = momentum / config.volatility + _RD_VALUE_NOISE * rng.standard_normal(n)
    rd_value[rng.random(n) < config.missing_join_rate] = np.nan

    future_close = np.concatenate(
        [close[_BARRIER_HORIZON:], np.repeat(close[-1], _BARRIER_HORIZON)]
    )
    signal_barrier = np.where(future_close >= close, 1, -1)

    features = dict(zip(FEATURE_COLUMNS, (rd_value, open_, high, low, close, volume), strict=True))
    frame = pd.DataFrame(
        {
            Column.TIMESTAMP: config.start_timestamp_ms + np.arange(n) * BAR_INTERVAL_MS,
            Column.SYMBOL: symbol,
            **features,
            Column.SIGNAL_BARRIER: signal_barrier,
        },
    )
    kept = rng.random(n) >= config.gap_rate
    return frame.loc[kept].reset_index(drop=True)


def _regime_drift(rng: np.random.Generator, config: SyntheticConfig) -> npt.NDArray[np.float64]:
    n = config.bars_per_symbol
    lengths = rng.geometric(1.0 / config.mean_regime_bars, size=n)
    n_regimes = int(np.searchsorted(np.cumsum(lengths), n)) + 1
    direction = np.where(np.arange(n_regimes) % 2 == 0, 1.0, -1.0) * rng.choice([-1.0, 1.0])
    magnitude = config.drift_scale * rng.uniform(0.5, 1.5, size=n_regimes)
    return np.repeat(direction * magnitude, lengths[:n_regimes])[:n]


def _plant_defects(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.astype(object)
    rows.loc[3, Column.OPEN] = "corrupt"
    rows.loc[6, Column.VOLUME] = -1.0
    rows.loc[9, [Column.HIGH, Column.LOW]] = rows.loc[9, [Column.LOW, Column.HIGH]].to_numpy()

    duplicate = rows.loc[[12]]
    conflicting = rows.loc[[15]].assign(**{Column.VOLUME: frame[Column.VOLUME].loc[[15]] * 2})
    out_of_order = [*range(18), 19, 18, *range(20, len(rows))]

    return pd.concat([rows.iloc[out_of_order], duplicate, conflicting], ignore_index=True)
