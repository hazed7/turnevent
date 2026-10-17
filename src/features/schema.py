from typing import Final

from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH
from core.frozen import FrozenModel
from features.builder import FEATURE_BUILDER_VERSION, FEATURES

NORMALIZATION: Final = (
    "stateless and per window, nothing is fitted on training data: prices enter as log "
    "ratios or fractions of the last close, volume as log1p ratios to the window mean, "
    "rd_value raw plus a z-score within the window"
)


class FeatureSpec(FrozenModel):
    name: str
    description: str


class FeatureSchema(FrozenModel):
    builder_version: str = FEATURE_BUILDER_VERSION
    window_length: int = WINDOW_LENGTH
    input_columns: tuple[str, ...] = FEATURE_COLUMNS
    row_order: str = "oldest_to_newest"
    normalization: str = NORMALIZATION
    features: tuple[FeatureSpec, ...]


def build_feature_schema() -> FeatureSchema:
    return FeatureSchema(
        features=tuple(
            FeatureSpec(
                name=f.name,
                description=f.description,
            )
            for f in FEATURES
        ),
    )
