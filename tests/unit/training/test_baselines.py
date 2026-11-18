import numpy as np
import pandas as pd
import pytest

from contracts.targets import TurnEvent
from training.baselines import MajorityBaseline, RdValueExtremumBaseline


def _rd(now: list[float], lag_1: list[float], lag_3: list[float], std: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rd_value": now,
            "rd_value_lag_1": lag_1,
            "rd_value_lag_3": lag_3,
            "rd_value_mean_20": [0.0] * len(now),
            "rd_value_std_20": [std] * len(now),
        }
    )


def _rule(
    reference_lag: int = 3, min_zscore: float = 1.0, min_rebound: float = 0.5, direction: int = 1
) -> RdValueExtremumBaseline:
    return RdValueExtremumBaseline.model_validate(
        {
            "reference_lag": reference_lag,
            "min_zscore": min_zscore,
            "min_rebound": min_rebound,
            "direction": direction,
        }
    )


def test_majority_predicts_the_most_frequent_training_class() -> None:
    baseline = MajorityBaseline.fit(np.array([0, 0, 1, -1, 0]))
    assert baseline.predict(_rd([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])).tolist() == [0, 0]


def test_rd_rule_needs_an_extreme_reference_and_a_rebound() -> None:
    features = _rd(
        now=[-1.0, 1.0, 0.5, -1.6, 0.0],
        lag_1=[0.0] * 5,
        lag_3=[-2.0, 2.0, -0.5, -2.0, 0.0],
    )
    assert _rule().predict(features).tolist() == [
        TurnEvent.TURN_UP,
        TurnEvent.TURN_DOWN,
        TurnEvent.NO_TURN,
        TurnEvent.NO_TURN,
        TurnEvent.NO_TURN,
    ]


def test_rd_rule_direction_can_invert_the_mapping() -> None:
    features = _rd(now=[-1.0, 1.0], lag_1=[0.0, 0.0], lag_3=[-2.0, 2.0])
    assert _rule(direction=-1).predict(features).tolist() == [
        TurnEvent.TURN_DOWN,
        TurnEvent.TURN_UP,
    ]


def test_rd_rule_can_use_the_previous_bar_as_reference() -> None:
    features = _rd(now=[-1.0], lag_1=[-2.0], lag_3=[0.0])
    assert _rule(reference_lag=1).predict(features).tolist() == [TurnEvent.TURN_UP]
    assert _rule(reference_lag=3).predict(features).tolist() == [TurnEvent.NO_TURN]


@pytest.mark.parametrize("std", [0.0, float("nan")])
def test_rd_rule_stays_silent_without_spread(std: float) -> None:
    features = _rd(now=[-1.0], lag_1=[0.0], lag_3=[-2.0], std=std)
    assert _rule().predict(features).tolist() == [TurnEvent.NO_TURN]


def test_rd_rule_parameters_are_chosen_by_macro_f1() -> None:
    features = _rd(
        now=[-1.0, 1.0, 0.0, 0.0],
        lag_1=[0.0] * 4,
        lag_3=[-2.0, 2.0, 0.0, 0.0],
    )
    truth = np.array([TurnEvent.TURN_DOWN, TurnEvent.TURN_UP, 0, 0])
    selected = RdValueExtremumBaseline.select(features, truth)
    assert selected.direction == -1
    assert selected.reference_lag == 3
    assert selected.predict(features).tolist() == truth.tolist()
