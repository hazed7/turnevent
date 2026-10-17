import numpy as np
import pandas as pd
import pytest

from training.metrics import (
    EVENT_NAMES,
    EvaluationConfig,
    classification_metrics,
    event_metrics,
    probability_metrics,
)

MINUTE = 60_000


def _keys(minutes: list[int], symbol: str = "A", lag: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": [m * MINUTE for m in minutes],
            "event_timestamp": [(m - lag) * MINUTE for m in minutes],
        },
    )


def test_classification_metrics_match_hand_counts() -> None:
    truth = np.array([-1, -1, 0, 0, 0, 1])
    predicted = np.array([-1, 0, 0, 0, 1, 1])
    metrics = classification_metrics(truth, predicted, EVENT_NAMES)
    assert metrics.confusion_matrix == ((1, 1, 0), (0, 2, 1), (0, 0, 1))
    assert metrics.per_class["TURN_UP"].precision == 0.5
    assert metrics.per_class["TURN_DOWN"].recall == 0.5
    assert metrics.balanced_accuracy == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert metrics.true_share["NO_TURN"] == 0.5


def test_false_turn_rate_is_the_share_of_wrong_turn_calls() -> None:
    truth = np.array([1, 0, 0, -1])
    predicted = np.array([1, 1, -1, 1])
    metrics = event_metrics(truth, predicted, _keys([1, 2, 3, 4]), EvaluationConfig())
    assert metrics.false_turn_rate == 0.75


def test_no_turn_calls_means_no_false_turn_rate() -> None:
    metrics = event_metrics(np.array([1, 0]), np.array([0, 0]), _keys([1, 2]), EvaluationConfig())
    assert metrics.false_turn_rate is None
    assert metrics.detection.detection_rate == 0.0


def test_detection_delay_counts_bars_to_the_first_matching_call() -> None:
    truth = np.array([1, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    predicted = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, -1])
    keys = _keys(list(range(10)))
    detection = event_metrics(
        truth, predicted, keys, EvaluationConfig(delay_tolerance_bars=3)
    ).detection
    assert (detection.true_turns, detection.detected) == (2, 1)
    assert detection.median_delay_bars == 2.0
    assert detection.detection_rate == 0.5


def test_calls_for_another_symbol_never_count_as_detection() -> None:
    keys = pd.concat([_keys([1], "A"), _keys([2], "B")], ignore_index=True)
    metrics = event_metrics(np.array([1, 0]), np.array([0, 1]), keys, EvaluationConfig())
    assert metrics.detection.detected == 0


def test_metrics_are_split_into_time_segments() -> None:
    truth = np.zeros(8, dtype=np.int64)
    metrics = event_metrics(truth, truth, _keys(list(range(8))), EvaluationConfig(time_segments=4))
    assert [segment.samples for segment in metrics.segments] == [2, 2, 2, 2]


def test_detection_delay_is_measured_from_the_event_bar() -> None:
    truth = np.array([1, 0, 0, 0])
    predicted = np.array([1, 0, 0, 0])
    detection = event_metrics(
        truth, predicted, _keys([4, 5, 6, 7], lag=3), EvaluationConfig()
    ).detection
    assert detection.median_delay_bars == 3.0


def test_probability_metrics_score_confident_mistakes_harshly() -> None:
    truth = np.array([1, 0])
    sure = pd.DataFrame([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], columns=[-1, 0, 1])
    wrong = pd.DataFrame([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], columns=[-1, 0, 1])
    perfect = probability_metrics(truth, sure, EVENT_NAMES)
    assert perfect.brier_score == 0.0
    assert perfect.log_loss == pytest.approx(0.0, abs=1e-12)
    assert probability_metrics(truth, wrong, EVENT_NAMES).brier_score == 1.0
    assert perfect.mean_probability == {"TURN_DOWN": 0.0, "NO_TURN": 0.5, "TURN_UP": 0.5}
    assert perfect.true_share == perfect.mean_probability
