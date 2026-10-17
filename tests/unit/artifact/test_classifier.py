import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from artifact.classifier import TurnClassifier, predict_codes
from core.errors import InvalidInputError


def _probabilities(*rows: tuple[float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[-1, 0, 1])


def test_without_threshold_the_most_likely_class_wins() -> None:
    probabilities = _probabilities((0.5, 0.3, 0.2), (0.1, 0.8, 0.1), (0.2, 0.3, 0.5))
    assert predict_codes(probabilities, None).tolist() == [-1, 0, 1]


def test_with_threshold_a_turn_needs_enough_probability() -> None:
    probabilities = _probabilities(
        (0.6, 0.3, 0.1), (0.1, 0.3, 0.6), (0.4, 0.2, 0.4), (0.3, 0.6, 0.1)
    )
    assert predict_codes(probabilities, 0.5).tolist() == [-1, 1, 0, 0]
    assert predict_codes(probabilities, 0.4).tolist() == [-1, 1, 1, 0]


def _dummy(classes: list[int]) -> DummyClassifier:
    return DummyClassifier(strategy="prior").fit(np.zeros((len(classes), 1)), classes)


def test_probabilities_follow_the_class_code_order() -> None:
    classifier = TurnClassifier(("x",), _dummy([1, 0, -1, -1]))
    probabilities = classifier.probabilities(pd.DataFrame({"x": [0.0]}))
    assert [int(code) for code in probabilities.columns] == [-1, 0, 1]
    assert probabilities.iloc[0].tolist() == [0.5, 0.25, 0.25]
    assert classifier.predict(pd.DataFrame({"x": [0.0]})).tolist() == [-1]


def test_estimator_must_know_every_class() -> None:
    with pytest.raises(InvalidInputError, match="all classes"):
        TurnClassifier(("x",), _dummy([0, 1]))


def test_feature_columns_must_match() -> None:
    classifier = TurnClassifier(("x",), _dummy([-1, 0, 1]))
    with pytest.raises(InvalidInputError, match="feature columns"):
        classifier.probabilities(pd.DataFrame({"y": [0.0]}))
