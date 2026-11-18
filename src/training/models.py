import math
from enum import StrEnum, auto
from typing import Annotated, Final, Self

import numpy as np
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
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


MAX_ALGORITHMS: Final = 2


class Algorithm(StrEnum):
    LOGISTIC_REGRESSION = auto()
    HIST_GRADIENT_BOOSTING = auto()


class BoostingConfig(FrozenModel):
    learning_rate: PositiveFloat = 0.05
    max_iter: PositiveInt = 300
    max_leaf_nodes: Annotated[int, Field(ge=2)] = 15
    min_samples_leaf: PositiveInt = 50
    l2_regularization: NonNegativeFloat = 1.0


class ModelConfig(FrozenModel):
    """The main algorithm comes first; the SoW allows at most one alternative.

    Every listed algorithm is fitted on train and the one with the best validation macro F1
    goes into the bundle. Ties keep the earlier algorithm.
    """

    algorithms: tuple[Algorithm, ...] = (
        Algorithm.LOGISTIC_REGRESSION,
        Algorithm.HIST_GRADIENT_BOOSTING,
    )
    regularization: PositiveFloat = 1.0
    max_iter: PositiveInt = 2_000
    boosting: BoostingConfig = BoostingConfig()
    calibrate: bool = True
    calibration_folds: Annotated[int, Field(ge=2)] = 3
    seed: int = 42

    @model_validator(mode="after")
    def _one_main_and_at_most_one_alternative(self) -> Self:
        if not 1 <= len(self.algorithms) <= MAX_ALGORITHMS:
            raise ValueError(f"list 1 to {MAX_ALGORITHMS} algorithms")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("algorithms must not repeat")
        return self


def fit_event_classifier(
    train: Samples,
    validation: Samples,
    config: ModelConfig,
    algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION,
) -> TurnClassifier:
    estimator = _fit(train, train.turn_event, config, algorithm)
    untuned = TurnClassifier(tuple(train.features.columns), estimator)
    probabilities = untuned.probabilities(validation.features)
    scores = [
        _macro_f1(validation.turn_event, predict_codes(probabilities, threshold))
        for threshold in THRESHOLD_GRID
    ]
    threshold = THRESHOLD_GRID[int(np.argmax(scores))]
    return TurnClassifier(untuned.feature_names, estimator, turn_threshold=threshold)


def fit_state_classifier(
    train: Samples,
    config: ModelConfig,
    algorithm: Algorithm = Algorithm.LOGISTIC_REGRESSION,
) -> TurnClassifier:
    estimator = _fit(train, train.turn_state, config, algorithm)
    return TurnClassifier(tuple(train.features.columns), estimator)


def _fit(train: Samples, target: Codes, config: ModelConfig, algorithm: Algorithm) -> Estimator:
    missing = set(CLASS_CODES) - set(np.unique(target).tolist())
    if missing:
        raise TrainingError(f"training data has no samples of classes {sorted(missing)}")
    model = _estimator(config, algorithm)
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


def _estimator(
    config: ModelConfig, algorithm: Algorithm
) -> Pipeline | HistGradientBoostingClassifier:
    if algorithm is Algorithm.HIST_GRADIENT_BOOSTING:
        boosting = config.boosting
        return HistGradientBoostingClassifier(
            learning_rate=boosting.learning_rate,
            max_iter=boosting.max_iter,
            max_leaf_nodes=boosting.max_leaf_nodes,
            min_samples_leaf=boosting.min_samples_leaf,
            l2_regularization=boosting.l2_regularization,
            class_weight="balanced",
            early_stopping=False,
            random_state=config.seed,
        )
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
