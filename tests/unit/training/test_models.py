import numpy as np
import pytest
from pydantic import ValidationError

from artifact.classifier import CLASS_CODES
from core.errors import TrainingError
from training.models import (
    THRESHOLD_GRID,
    Algorithm,
    ModelConfig,
    fit_event_classifier,
    fit_state_classifier,
)
from training.samples import Samples
from training.split import Part, SplitConfig, time_split


@pytest.fixture(scope="module")
def parts(samples: Samples) -> dict[Part, Samples]:
    return dict(time_split(samples, SplitConfig())[0])


def test_event_threshold_comes_from_the_grid(parts: dict[Part, Samples]) -> None:
    classifier = fit_event_classifier(parts[Part.TRAIN], parts[Part.VALIDATION], ModelConfig())
    assert classifier.turn_threshold in THRESHOLD_GRID
    probabilities = classifier.probabilities(parts[Part.TEST].features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_state_classifier_uses_plain_argmax(parts: dict[Part, Samples]) -> None:
    classifier = fit_state_classifier(parts[Part.TRAIN], ModelConfig())
    assert classifier.turn_threshold is None
    assert set(classifier.predict(parts[Part.TEST].features)) <= set(CLASS_CODES)


def test_training_without_a_class_raises(parts: dict[Part, Samples]) -> None:
    train = parts[Part.TRAIN]
    no_turns = train.take(train.turn_event == 0)
    with pytest.raises(TrainingError, match="no samples of classes"):
        fit_event_classifier(no_turns, parts[Part.VALIDATION], ModelConfig())


def test_calibration_brings_turn_probabilities_to_their_true_rate(
    parts: dict[Part, Samples],
) -> None:
    train, validation = parts[Part.TRAIN], parts[Part.VALIDATION]
    true_rate = float((validation.turn_event != 0).mean())

    def mean_turn_probability(*, calibrate: bool) -> float:
        classifier = fit_event_classifier(train, validation, ModelConfig(calibrate=calibrate))
        probabilities = classifier.probabilities(validation.features)
        return float((probabilities[-1] + probabilities[1]).mean())

    raw = mean_turn_probability(calibrate=False)
    calibrated = mean_turn_probability(calibrate=True)
    assert abs(calibrated - true_rate) < abs(raw - true_rate)


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_every_algorithm_gives_calibrated_probabilities(
    parts: dict[Part, Samples], algorithm: Algorithm, fast_model: ModelConfig
) -> None:
    train, validation = parts[Part.TRAIN], parts[Part.VALIDATION]
    event = fit_event_classifier(train, validation, fast_model, algorithm)
    state = fit_state_classifier(train, fast_model, algorithm)
    assert event.turn_threshold in THRESHOLD_GRID
    for classifier in (event, state):
        probabilities = classifier.probabilities(parts[Part.TEST].features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_boosting_is_deterministic(parts: dict[Part, Samples], fast_model: ModelConfig) -> None:
    train, validation = parts[Part.TRAIN], parts[Part.VALIDATION]
    algorithm = Algorithm.HIST_GRADIENT_BOOSTING
    first = fit_event_classifier(train, validation, fast_model, algorithm)
    again = fit_event_classifier(train, validation, fast_model, algorithm)
    features = parts[Part.TEST].features
    np.testing.assert_array_equal(first.probabilities(features), again.probabilities(features))


@pytest.mark.parametrize(
    "algorithms",
    [
        (),
        (Algorithm.LOGISTIC_REGRESSION, Algorithm.LOGISTIC_REGRESSION),
        (
            Algorithm.LOGISTIC_REGRESSION,
            Algorithm.HIST_GRADIENT_BOOSTING,
            Algorithm.LOGISTIC_REGRESSION,
        ),
    ],
)
def test_config_allows_one_main_and_at_most_one_alternative(
    algorithms: tuple[Algorithm, ...],
) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(algorithms=algorithms)
