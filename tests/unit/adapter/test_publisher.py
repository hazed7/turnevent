import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from adapter.ledger import PublicationLedger
from adapter.publisher import Mode, OutboxFile, Outcome, Publisher
from artifact.bundle import LoadedBundle, load_bundle
from contracts.payload import SignalPayload, Source
from contracts.targets import TurnEvent
from inference.results import InferenceBatch, WindowResult, WindowStatus
from inference.runner import run_inference

type MakeResult = Callable[..., WindowResult]


def _batch(*results: WindowResult) -> InferenceBatch:
    return InferenceBatch(
        generated_at=None, model_version="tm-test", envelope_error=None, results=results
    )


def _outbox(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _shadow(tmp_path: Path) -> Publisher:
    ledger = PublicationLedger(tmp_path / "ledger.jsonl")
    return Publisher(Mode.SHADOW, ledger, OutboxFile(tmp_path / "outbox.jsonl"))


def test_shadow_mode_sends_only_confirmed_turns(make_result: MakeResult, tmp_path: Path) -> None:
    batch = _batch(
        make_result(TurnEvent.TURN_UP, symbol="UP"),
        make_result(TurnEvent.NO_TURN, symbol="FLAT"),
        make_result(status=WindowStatus.NOT_READY, symbol="WAIT"),
        make_result(TurnEvent.TURN_DOWN, symbol="DOWN"),
    )
    decisions = _shadow(tmp_path).publish(batch)

    assert [d.outcome for d in decisions] == [
        Outcome.PUBLISHED,
        Outcome.NOT_PUBLISHABLE,
        Outcome.NOT_PUBLISHABLE,
        Outcome.PUBLISHED,
    ]
    sent = _outbox(tmp_path / "outbox.jsonl")
    assert [(p["symbol"], p["signal"], p["source"]) for p in sent] == [
        ("UP", "BUY", "shadow"),
        ("DOWN", "SELL", "shadow"),
    ]


def test_dry_run_sends_and_records_nothing(make_result: MakeResult, tmp_path: Path) -> None:
    ledger = PublicationLedger(tmp_path / "ledger.jsonl")
    publisher = Publisher(Mode.DRY_RUN, ledger, OutboxFile(tmp_path / "outbox.jsonl"))
    (decision,) = publisher.publish(_batch(make_result()))
    assert decision.outcome is Outcome.WOULD_PUBLISH
    assert decision.payload is not None
    assert decision.payload.source is Source.ML
    assert _outbox(tmp_path / "outbox.jsonl") == []
    assert not (tmp_path / "ledger.jsonl").exists()


def test_a_window_is_never_sent_twice(make_result: MakeResult, tmp_path: Path) -> None:
    batch = _batch(make_result(), make_result())
    first = _shadow(tmp_path).publish(batch)
    again = _shadow(tmp_path).publish(batch)
    assert [d.outcome for d in first] == [Outcome.PUBLISHED, Outcome.DUPLICATE]
    assert [d.outcome for d in again] == [Outcome.DUPLICATE, Outcome.DUPLICATE]
    assert len(_outbox(tmp_path / "outbox.jsonl")) == 1


class _Decline:
    def allows(self, payload: SignalPayload) -> bool:
        return payload.signal != "SELL"


def test_decision_layer_can_hold_back_a_transition(make_result: MakeResult, tmp_path: Path) -> None:
    publisher = Publisher(
        Mode.SHADOW,
        PublicationLedger(),
        OutboxFile(tmp_path / "outbox.jsonl"),
        decision_layer=_Decline(),
    )
    decisions = publisher.publish(_batch(make_result(TurnEvent.TURN_DOWN)))
    assert decisions[0].outcome is Outcome.DECLINED
    assert _outbox(tmp_path / "outbox.jsonl") == []


class _Unreachable:
    def send(self, payload: SignalPayload) -> None:
        raise ConnectionError(payload.symbol)


def test_failed_send_is_not_recorded_so_a_retry_is_not_a_duplicate(
    make_result: MakeResult,
) -> None:
    ledger = PublicationLedger()
    with pytest.raises(ConnectionError):
        Publisher(Mode.SHADOW, ledger, _Unreachable()).publish(
            _batch(make_result(window_end=60_000))
        )
    assert ("SYM001", 60_000) not in ledger


def test_decisions_carry_the_telemetry_fields(make_result: MakeResult) -> None:
    publisher = Publisher(Mode.DRY_RUN, PublicationLedger(), _Unreachable())
    predicted, waiting = publisher.publish(
        _batch(make_result(), make_result(status=WindowStatus.NOT_READY))
    )
    assert predicted.predicted_turn_event == "TURN_UP"
    assert predicted.predicted_turn_state == "BUY_HOLD"
    assert predicted.event_probabilities is not None
    assert predicted.model_version == "tm-test"
    assert waiting.status is WindowStatus.NOT_READY
    assert waiting.reason == "stale_data"
    assert waiting.payload is None


def test_fixture_runs_end_to_end_without_publishing_errors(
    trained_bundle_dir: Path, load_fixture: Callable[[str], Any], tmp_path: Path
) -> None:
    bundle: LoadedBundle = load_bundle(trained_bundle_dir)
    envelope = load_fixture("online-ready.json")
    envelope["items"] += load_fixture("online-invalid.json")["items"]
    batch = run_inference(json.dumps(envelope), bundle)
    decisions = _shadow(tmp_path).publish(batch)
    sent = {p["symbol"] for p in _outbox(tmp_path / "outbox.jsonl")}
    assert len(decisions) == len(envelope["items"])
    assert not any(symbol.startswith("BAD") for symbol in sent)
    for decision in decisions:
        if decision.status is not WindowStatus.PREDICTED:
            assert decision.outcome is Outcome.NOT_PUBLISHABLE
