from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from artifact.classifier import Codes
from contracts.targets import TurnEvent

RD_RULE_CHANGE_GRID: Final = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class MajorityBaseline:
    code: int

    @classmethod
    def fit(cls, labels: Codes) -> "MajorityBaseline":
        codes, counts = np.unique(labels, return_counts=True)
        return cls(code=int(codes[counts.argmax()]))

    def predict(self, features: pd.DataFrame) -> Codes:
        return np.full(len(features), self.code, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class RdValueCrossBaseline:
    min_change: float

    @classmethod
    def select(cls, features: pd.DataFrame, labels: Codes) -> "RdValueCrossBaseline":
        scores = [
            f1_score(labels, cls(change).predict(features), average="macro", zero_division=0)
            for change in RD_RULE_CHANGE_GRID
        ]
        return cls(min_change=RD_RULE_CHANGE_GRID[int(np.argmax(scores))])

    def predict(self, features: pd.DataFrame) -> Codes:
        now = features["rd_value"].to_numpy()
        before = features["rd_value_lag_1"].to_numpy()
        jump = np.abs(now - before) >= self.min_change
        up = (before < 0) & (now >= 0) & jump
        down = (before > 0) & (now <= 0) & jump
        return np.select([up, down], [TurnEvent.TURN_UP, TurnEvent.TURN_DOWN], TurnEvent.NO_TURN)
