import math
from typing import Annotated, Final

import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from artifact.classifier import CLASS_CODES, Codes, Estimator, TurnClassifier, predict_codes
from core.errors import TrainingError
from core.frozen import FrozenModel
from training.samples import SampleColumn, Samples

THRESHOLD_GRID: Final = tuple(round(0.05 + 0.01 * step, 2) for step in range(95))


class ModelConfig(FrozenModel):
    regularization: PositiveFloat = 1.0
    max_iter: PositiveInt = 2_000
    calibrate: bool = True
    calibration_folds: Annotated[int, Field(ge=2)] = 3
    seed: int = 42


def fit_event_classifier(
    train: Samples, validation: Samples, config: ModelConfig
) -> TurnClassifier:
    estimator = _fit(train, train.turn_event, config)
    untuned = TurnClassifier(tuple(train.features.columns), estimator)
    probabilities = untuned.probabilities(validation.features)
    scores = [
        _macro_f1(validation.turn_event, predict_codes(probabilities, threshold))
        for threshold in THRESHOLD_GRID
    ]
    threshold = THRESHOLD_GRID[int(np.argmax(scores))]
    return TurnClassifier(untuned.feature_names, estimator, turn_threshold=threshold)


def fit_state_classifier(train: Samples, config: ModelConfig) -> TurnClassifier:
    return TurnClassifier(tuple(train.features.columns), _fit(train, train.turn_state, config))


def _fit(train: Samples, target: Codes, config: ModelConfig) -> Estimator:
    missing = set(CLASS_CODES) - set(np.unique(target).tolist())
    if missing:
        raise TrainingError(f"training data has no samples of classes {sorted(missing)}")
    model = _estimator(config)
    if config.calibrate:
        # Each fold calibrates on later samples than it trained on, with a purge gap between.
        folds = TimeSeriesSplit(n_splits=config.calibration_folds, gap=_purge_gap(train))
        model = CalibratedClassifierCV(model, method="isotonic", cv=folds)
    model.fit(train.features, target)
    fitted: Estimator = model
    return fitted


def _purge_gap(train: Samples) -> int:
    span = train.keys[SampleColumn.SPAN_END_MS] - train.keys[SampleColumn.SPAN_START_MS]
    timestamps = train.keys["timestamp"]
    duration = max(int(timestamps.max() - timestamps.min()), 1)
    return math.ceil(len(train) * int(span.max()) / duration)


def _estimator(config: ModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classify",
                LogisticRegression(
                    C=config.regularization,
                    class_weight="balanced",
                    max_iter=config.max_iter,
                    random_state=config.seed,
                ),
            ),
        ],
    )


def _macro_f1(truth: Codes, predicted: Codes) -> float:
    return float(f1_score(truth, predicted, labels=CLASS_CODES, average="macro", zero_division=0))
