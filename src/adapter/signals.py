from typing import Final

from contracts.payload import Signal, SignalPayload, Source
from contracts.targets import TurnEvent, TurnState
from inference.results import WindowResult, WindowStatus

SIGNAL_BY_TURN: Final = {TurnEvent.TURN_UP: Signal.BUY, TurnEvent.TURN_DOWN: Signal.SELL}
RATING_STATES: Final = (TurnState.SELL_HOLD.name, TurnState.BUY_HOLD.name)


class NotPublishableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def build_payload(result: WindowResult, source: Source) -> SignalPayload:
    if result.status is not WindowStatus.PREDICTED:
        raise NotPublishableError(f"{result.status}: {result.reason}")
    if result.prediction is None or result.last_bar is None or result.symbol is None:
        raise NotPublishableError("prediction, last bar or symbol is missing")
    if result.window_end_timestamp is None:
        raise NotPublishableError("windowEndTimestamp is missing")
    signal = SIGNAL_BY_TURN.get(result.prediction.turn_event)
    if signal is None:
        raise NotPublishableError("NO_TURN never produces a signal")
    return SignalPayload(
        symbol=result.symbol,
        timestamp=result.window_end_timestamp,
        signal=signal,
        price=result.last_bar.close,
        rating=max(result.prediction.state_probabilities[state] for state in RATING_STATES),
        source=source,
    )
