import json
import os
from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from typing import Protocol

from adapter.ledger import PublicationLedger
from adapter.signals import NotPublishableError, build_payload
from contracts.payload import SignalPayload, Source
from core.frozen import FrozenModel
from inference.results import InferenceBatch, WindowResult, WindowStatus


class Mode(StrEnum):
    DRY_RUN = auto()
    SHADOW = auto()


class Outcome(StrEnum):
    PUBLISHED = auto()
    WOULD_PUBLISH = auto()
    DUPLICATE = auto()
    DECLINED = auto()
    NOT_PUBLISHABLE = auto()


class Decision(FrozenModel):
    symbol: str | None
    window_end_timestamp: int | None
    status: WindowStatus
    reason: str | None
    predicted_turn_event: str | None
    event_probabilities: Mapping[str, float] | None
    predicted_turn_state: str | None
    state_probabilities: Mapping[str, float] | None
    model_version: str
    feature_schema_version: str
    outcome: Outcome
    outcome_reason: str | None
    payload: SignalPayload | None


class DecisionLayer(Protocol):
    def allows(self, payload: SignalPayload) -> bool: ...


class Sink(Protocol):
    def send(self, payload: SignalPayload) -> None: ...


class PassThrough:
    def allows(self, payload: SignalPayload) -> bool:  # noqa: ARG002
        return True


class OutboxFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, payload: SignalPayload) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload.model_dump(mode="json"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class Publisher:
    def __init__(
        self,
        mode: Mode,
        ledger: PublicationLedger,
        sink: Sink,
        decision_layer: DecisionLayer | None = None,
    ) -> None:
        self.mode = mode
        self.ledger = ledger
        self.sink = sink
        self.decision_layer = decision_layer or PassThrough()

    @property
    def source(self) -> Source:
        return Source.SHADOW if self.mode is Mode.SHADOW else Source.ML

    def publish(self, batch: InferenceBatch) -> tuple[Decision, ...]:
        return tuple(self._decide(result) for result in batch.results)

    def _decide(self, result: WindowResult) -> Decision:
        try:
            payload = build_payload(result, self.source)
        except NotPublishableError as skipped:
            return _decision(result, Outcome.NOT_PUBLISHABLE, skipped.reason, None)
        key = (payload.symbol, payload.timestamp)
        if key in self.ledger:
            return _decision(result, Outcome.DUPLICATE, "window was already published", None)
        if not self.decision_layer.allows(payload):
            return _decision(
                result, Outcome.DECLINED, "decision layer declined the transition", None
            )
        if self.mode is Mode.DRY_RUN:
            return _decision(result, Outcome.WOULD_PUBLISH, None, payload)
        self.sink.send(payload)
        self.ledger.record(key)
        return _decision(result, Outcome.PUBLISHED, None, payload)


def _decision(
    result: WindowResult, outcome: Outcome, reason: str | None, payload: SignalPayload | None
) -> Decision:
    prediction = result.prediction
    return Decision(
        symbol=result.symbol,
        window_end_timestamp=result.window_end_timestamp,
        status=result.status,
        reason=result.reason,
        predicted_turn_event=prediction.turn_event.name if prediction else None,
        event_probabilities=prediction.event_probabilities if prediction else None,
        predicted_turn_state=prediction.turn_state.name if prediction else None,
        state_probabilities=prediction.state_probabilities if prediction else None,
        model_version=result.model_version,
        feature_schema_version=result.feature_schema_version,
        outcome=outcome,
        outcome_reason=reason,
        payload=payload,
    )
