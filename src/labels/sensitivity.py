from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from core.frozen import FrozenModel
from dataset.cleaning import CleanDataset
from labels.definition import LabelSummary, summarize_labels
from labels.generator import LabelConfig, generate_labels

type Variations = Mapping[str, tuple[int | float, ...]]

DEFAULT_VARIATIONS: Final[Variations] = MappingProxyType(
    {
        "threshold_coefficient": (0.5, 0.65, 0.8, 1.0),
        "min_separation_bars": (2, 4, 6),
        "left_bars": (2, 3, 4),
        "right_bars": (2, 3, 4),
    },
)


class SensitivityRow(FrozenModel):
    parameter: str
    value: int | float
    summary: LabelSummary


def label_sensitivity(
    clean: CleanDataset, base: LabelConfig, variations: Variations = DEFAULT_VARIATIONS
) -> tuple[SensitivityRow, ...]:
    return tuple(
        SensitivityRow(
            parameter=parameter,
            value=value,
            summary=summarize_labels(generate_labels(clean, _vary(base, parameter, value))),
        )
        for parameter, values in variations.items()
        for value in values
    )


def _vary(base: LabelConfig, parameter: str, value: float) -> LabelConfig:
    return LabelConfig.model_validate({**base.model_dump(), parameter: value})
