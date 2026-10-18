from collections.abc import Callable

import pytest

from adapter.signals import NotPublishableError, build_payload
from contracts.payload import Signal, Source
from contracts.targets import TurnEvent
from inference.results import WindowResult, WindowStatus

type MakeResult = Callable[..., WindowResult]


@pytest.mark.parametrize(
    ("turn", "signal"), [(TurnEvent.TURN_UP, Signal.BUY), (TurnEvent.TURN_DOWN, Signal.SELL)]
)
def test_confirmed_turns_become_signals(
    make_result: MakeResult, turn: TurnEvent, signal: Signal
) -> None:
    payload = build_payload(make_result(turn, window_end=123_000), Source.ML)
    assert payload.signal is signal
    assert payload.symbol == "SYM001"
    assert payload.timestamp == 123_000
    assert payload.price == 101.5
    assert payload.source is Source.ML


def test_rating_is_the_strongest_hold_state_never_neutral(make_result: MakeResult) -> None:
    result = make_result(state_probabilities=(0.06, 0.9, 0.04))
    assert build_payload(result, Source.SHADOW).rating == 0.06


def test_no_turn_never_produces_a_signal(make_result: MakeResult) -> None:
    with pytest.raises(NotPublishableError, match="NO_TURN"):
        build_payload(make_result(TurnEvent.NO_TURN), Source.ML)


@pytest.mark.parametrize(
    "status", [WindowStatus.NOT_READY, WindowStatus.INVALID_INPUT, WindowStatus.MODEL_ERROR]
)
def test_errors_and_unready_windows_never_produce_a_signal(
    make_result: MakeResult, status: WindowStatus
) -> None:
    with pytest.raises(NotPublishableError, match=str(status)):
        build_payload(make_result(status=status), Source.ML)


def test_results_without_symbol_or_window_end_are_not_published(make_result: MakeResult) -> None:
    no_symbol = make_result().model_copy(update={"symbol": None})
    no_end = make_result().model_copy(update={"window_end_timestamp": None})
    with pytest.raises(NotPublishableError, match="symbol"):
        build_payload(no_symbol, Source.ML)
    with pytest.raises(NotPublishableError, match="windowEndTimestamp"):
        build_payload(no_end, Source.ML)
