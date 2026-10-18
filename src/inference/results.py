from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Self

from pydantic import AwareDatetime, model_validator

from contracts.targets import TurnEvent, TurnState
from core.frozen import FrozenModel


class WindowStatus(StrEnum):
    PREDICTED = auto()
    NOT_READY = auto()
    INVALID_INPUT = auto()
    MODEL_ERROR = auto()


class LastBar(FrozenModel):
    close: float
    rd_value: float
    volume: float


class Prediction(FrozenModel):
    turn_event: TurnEvent
    event_probabilities: Mapping[str, float]
    turn_state: TurnState
    state_probabilities: Mapping[str, float]


class WindowResult(FrozenModel):
    symbol: str | None
    window_end_timestamp: int | None
    status: WindowStatus
    reason: str | None
    prediction: Prediction | None
    last_bar: LastBar | None
    model_version: str
    feature_schema_version: str

    @model_validator(mode="after")
    def _prediction_iff_predicted(self) -> Self:
        predicted = self.status is WindowStatus.PREDICTED

        if predicted != (self.prediction is not None and self.last_bar is not None):
            raise ValueError("a prediction and last bar exist exactly when status is PREDICTED")
        if predicted == (self.reason is not None):
            raise ValueError("a reason is given exactly when there is no prediction")
        return self


class InferenceBatch(FrozenModel):
    generated_at: AwareDatetime | None
    model_version: str
    envelope_error: str | None
    results: tuple[WindowResult, ...]
