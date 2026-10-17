from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Annotated, Final, Self

import numpy as np
from pydantic import Field, NonNegativeInt, model_validator

from contracts.market import BAR_INTERVAL_MS, Column
from core.errors import TrainingError
from core.frozen import FrozenModel
from core.serialization import fingerprint
from training.samples import SampleColumn, Samples

SPLIT_POLICY_VERSION: Final = "1.0.0"

type Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]


class Part(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class SplitConfig(FrozenModel):
    validation_fraction: Fraction = 0.2
    test_fraction: Fraction = 0.2
    embargo_bars: NonNegativeInt = 60

    @model_validator(mode="after")
    def _leaves_room_to_train(self) -> Self:
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions must leave data to train on")
        return self


class Period(FrozenModel):
    part: Part
    start_ms: int
    end_ms: int
    samples: int
    dropped_at_edges: int
    keys_fingerprint: str


class SplitManifest(FrozenModel):
    policy_version: str = SPLIT_POLICY_VERSION
    config: SplitConfig
    boundaries_ms: tuple[int, int]
    embargo_ms: int
    periods: tuple[Period, ...]
    rule: str = (
        "sequential by event time, never shuffled; a sample belongs to a part only if every bar "
        "its features or label depend on lies inside that part's period (purge), and each "
        "later period starts embargo_ms after its boundary (embargo)"
    )


def time_split(
    samples: Samples, config: SplitConfig, *, interval_ms: int = BAR_INTERVAL_MS
) -> tuple[Mapping[Part, Samples], SplitManifest]:
    timestamps = samples.keys[Column.TIMESTAMP].to_numpy()
    span_start = samples.keys[SampleColumn.SPAN_START_MS].to_numpy()
    span_end = samples.keys[SampleColumn.SPAN_END_MS].to_numpy()

    train_end = int(np.quantile(timestamps, 1 - config.validation_fraction - config.test_fraction))
    validation_end = int(np.quantile(timestamps, 1 - config.test_fraction))
    embargo_ms = config.embargo_bars * interval_ms
    edges = {
        Part.TRAIN: (int(span_start.min()), train_end),
        Part.VALIDATION: (train_end + embargo_ms, validation_end),
        Part.TEST: (validation_end + embargo_ms, int(span_end.max()) + 1),
    }
    owners = {
        Part.TRAIN: timestamps < train_end,
        Part.VALIDATION: (timestamps >= train_end) & (timestamps < validation_end),
        Part.TEST: timestamps >= validation_end,
    }

    parts, periods = {}, []
    for part, (start, end) in edges.items():
        inside = (span_start >= start) & (span_end < end)
        chosen = samples.take(inside)
        if not len(chosen):
            raise TrainingError(f"{part} split is empty; use more data or a smaller embargo")
        parts[part] = chosen
        periods.append(
            Period(
                part=part,
                start_ms=start,
                end_ms=end,
                samples=len(chosen),
                dropped_at_edges=int((owners[part] & ~inside).sum()),
                keys_fingerprint=fingerprint(
                    chosen.keys[[Column.SYMBOL, Column.TIMESTAMP]].to_numpy().tolist()
                ),
            ),
        )

    manifest = SplitManifest(
        config=config,
        boundaries_ms=(train_end, validation_end),
        embargo_ms=embargo_ms,
        periods=tuple(periods),
    )
    return parts, manifest
