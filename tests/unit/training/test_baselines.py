import numpy as np
import pandas as pd

from contracts.targets import TurnEvent
from training.baselines import MajorityBaseline, RdValueCrossBaseline


def _rd(now: list[float], before: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"rd_value": now, "rd_value_lag_1": before})


def test_majority_predicts_the_most_frequent_training_class() -> None:
    baseline = MajorityBaseline.fit(np.array([0, 0, 1, -1, 0]))
    assert baseline.predict(_rd([0.0, 0.0], [0.0, 0.0])).tolist() == [0, 0]


def test_rd_rule_fires_on_zero_crossings_with_a_big_enough_jump() -> None:
    features = _rd(now=[0.5, -0.5, 0.1, 0.5, -2.0], before=[-0.5, 0.5, -0.05, 0.6, 1.0])
    assert RdValueCrossBaseline(min_change=0.5).predict(features).tolist() == [
        TurnEvent.TURN_UP,
        TurnEvent.TURN_DOWN,
        TurnEvent.NO_TURN,
        TurnEvent.NO_TURN,
        TurnEvent.TURN_DOWN,
    ]


def test_rd_rule_threshold_is_chosen_by_macro_f1() -> None:
    features = _rd(now=[0.1, 2.0, -2.0, 0.1], before=[-0.1, -1.0, 1.0, -0.1])
    truth = np.array([0, 1, -1, 0])
    assert RdValueCrossBaseline.select(features, truth).min_change == 0.25
