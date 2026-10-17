from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from core.errors import InvalidInputError

CLASS_CODES: Final = (-1, 0, 1)
ABSTAIN_CODE: Final = 0

type Codes = npt.NDArray[np.int64]


class Estimator(Protocol):
    classes_: Any

    def predict_proba(self, features: Any) -> Any: ...  # noqa: ANN401


@dataclass(frozen=True, slots=True)
class TurnClassifier:
    feature_names: tuple[str, ...]
    estimator: Estimator
    turn_threshold: float | None = None

    def __post_init__(self) -> None:
        if tuple(int(code) for code in self.estimator.classes_) != CLASS_CODES:
            raise InvalidInputError(f"estimator must be fitted on all classes {CLASS_CODES}")

    def probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        if tuple(features.columns) != self.feature_names:
            raise InvalidInputError("feature columns do not match the classifier's feature names")
        values = np.asarray(self.estimator.predict_proba(features), dtype=np.float64)
        return pd.DataFrame(values, columns=list(CLASS_CODES), index=features.index)

    def predict(self, features: pd.DataFrame) -> Codes:
        return predict_codes(self.probabilities(features), self.turn_threshold)


def predict_codes(probabilities: pd.DataFrame, turn_threshold: float | None) -> Codes:
    values = probabilities[list(CLASS_CODES)].to_numpy()
    if turn_threshold is None:
        return np.asarray(CLASS_CODES, dtype=np.int64)[values.argmax(axis=1)]
    down, up = values[:, 0], values[:, 2]
    strongest = np.where(up >= down, CLASS_CODES[2], CLASS_CODES[0])
    return np.where(np.maximum(up, down) >= turn_threshold, strongest, ABSTAIN_CODE)
