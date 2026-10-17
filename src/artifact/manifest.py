from collections.abc import Mapping
from typing import Final

from contracts.market import TIMEFRAME, WINDOW_LENGTH
from core.frozen import FrozenModel
from core.runtime import RuntimeInfo
from labels.definition import ClassDefinition

ARTIFACT_KIND: Final = "turnevent-classifier-bundle"
BUNDLE_FORMAT_VERSION: Final = "1.0.0"


class ModelFile(FrozenModel):
    path: str
    sha256: str


class MetricsSummary(FrozenModel):
    test_samples: int
    macro_f1: float
    balanced_accuracy: float
    false_turn_rate: float | None
    detection_rate: float | None
    median_delay_bars: float | None
    majority_macro_f1: float
    rd_value_rule_macro_f1: float
    state_macro_f1: float
    event_log_loss: float
    state_log_loss: float


class ModelManifest(FrozenModel):
    artifact_kind: str = ARTIFACT_KIND
    bundle_format_version: str = BUNDLE_FORMAT_VERSION
    model_version: str
    dataset_version: str
    dataset_sha256: str
    feature_schema_version: str
    label_schema_version: str
    label_generator_version: str
    timeframe: str = TIMEFRAME
    window_length: int = WINDOW_LENGTH
    prediction_lag_bars: int
    normalization_spec: str
    feature_names: tuple[str, ...]
    event_classes: tuple[ClassDefinition, ...]
    state_classes: tuple[ClassDefinition, ...]
    event_turn_threshold: float
    event_model: ModelFile
    state_model: ModelFile
    documents: Mapping[str, str]
    metrics: MetricsSummary
    runtime: RuntimeInfo
