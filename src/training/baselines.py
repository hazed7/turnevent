from dataclasses import dataclass
from itertools import product
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from artifact.classifier import Codes
from contracts.targets import TurnEvent
from core.frozen import FrozenModel

RD_RULE_REFERENCE_LAGS: Final[tuple[Literal[1, 3], ...]] = (1, 3)
RD_RULE_ZSCORE_GRID: Final = (0.5, 1.0, 1.5, 2.0)
RD_RULE_REBOUND_GRID: Final = (0.25, 0.5, 1.0)
RD_RULE_DIRECTIONS: Final[tuple[Literal[1, -1], ...]] = (1, -1)


@dataclass(frozen=True, slots=True)
class MajorityBaseline:
    code: int

    @classmethod
    def fit(cls, labels: Codes) -> "MajorityBaseline":
        codes, counts = np.unique(labels, return_counts=True)
        return cls(code=int(codes[counts.argmax()]))

    def predict(self, features: pd.DataFrame) -> Codes:
        return np.full(len(features), self.code, dtype=np.int64)


class RdValueExtremumBaseline(FrozenModel):
    """Turn when rd_value had an extreme z-score a few bars ago and has since reversed.

    Uses only causal window features: the reference bar is ``reference_lag`` bars back, its
    z-score is measured against the last 20 bars, and the rebound since then is scaled by the
    same standard deviation. ``direction`` maps an rd_value low to TURN_UP (1) or TURN_DOWN (-1).
    All four parameters are chosen on validation.
    """

    reference_lag: Literal[1, 3]
    min_zscore: float
    min_rebound: float
    direction: Literal[1, -1]

    @classmethod
    def select(cls, features: pd.DataFrame, labels: Codes) -> "RdValueExtremumBaseline":
        candidates = [
            cls(reference_lag=lag, min_zscore=z, min_rebound=r, direction=d)
            for lag, z, r, d in product(
                RD_RULE_REFERENCE_LAGS,
                RD_RULE_ZSCORE_GRID,
                RD_RULE_REBOUND_GRID,
                RD_RULE_DIRECTIONS,
            )
        ]
        scores = [
            f1_score(labels, rule.predict(features), average="macro", zero_division=0)
            for rule in candidates
        ]
        return candidates[int(np.argmax(scores))]

    def predict(self, features: pd.DataFrame) -> Codes:
        now = features["rd_value"].to_numpy()
        reference = features[f"rd_value_lag_{self.reference_lag}"].to_numpy()
        mean = features["rd_value_mean_20"].to_numpy()
        std = features["rd_value_std_20"].to_numpy()

        valid = np.isfinite(std) & (std > 0)
        scale = np.where(valid, std, 1.0)
        zscore = (reference - mean) / scale
        rebound = (now - reference) / scale

        low = valid & (zscore <= -self.min_zscore) & (rebound >= self.min_rebound)
        high = valid & (zscore >= self.min_zscore) & (rebound <= -self.min_rebound)
        up, down = (low, high) if self.direction == 1 else (high, low)
        return np.select([up, down], [TurnEvent.TURN_UP, TurnEvent.TURN_DOWN], TurnEvent.NO_TURN)
