import json
from collections.abc import Callable
from typing import Any

import pytest

from contracts.online import (
    NotReadyReason,
    PendingWindow,
    ReadyWindow,
    RejectedWindow,
    parse_envelope,
)
from core.errors import InvalidInputError

type LoadFixture = Callable[[str], Any]


def _parse(envelope: dict[str, Any]) -> tuple[ReadyWindow | PendingWindow | RejectedWindow, ...]:
    return parse_envelope(json.dumps(envelope)).items


def test_ready_windows_parse_into_64x6_matrices(load_fixture: LoadFixture) -> None:
    items = _parse(load_fixture("online-ready.json"))
    assert [type(item) for item in items] == [ReadyWindow, ReadyWindow]
    for item in items:
        assert isinstance(item, ReadyWindow)
        assert len(item.features) == 64
        assert {len(row) for row in item.features} == {6}


def test_gaps_inside_a_ready_window_are_allowed(load_fixture: LoadFixture) -> None:
    gapped = _parse(load_fixture("online-ready.json"))[1]
    assert isinstance(gapped, ReadyWindow)
    assert gapped.window_end_timestamp - gapped.window_start_timestamp > 63 * 60_000


def test_warmup_windows_carry_no_features(load_fixture: LoadFixture) -> None:
    items = _parse(load_fixture("online-warmup.json"))
    assert all(isinstance(item, PendingWindow) and item.features is None for item in items)
    assert [item.reason for item in items if isinstance(item, PendingWindow)] == [
        NotReadyReason.NOT_ENOUGH_POINTS,
        NotReadyReason.STALE_DATA,
        NotReadyReason.MISSING_JOINED_ROWS,
        None,
    ]


def test_every_invalid_item_is_rejected_with_its_symbol(load_fixture: LoadFixture) -> None:
    items = _parse(load_fixture("online-invalid.json"))
    assert all(isinstance(item, RejectedWindow) for item in items)
    assert [item.symbol for item in items] == [f"BAD00{n}" for n in range(1, 8)]


def test_a_bad_item_does_not_spoil_its_neighbours(load_fixture: LoadFixture) -> None:
    envelope = load_fixture("online-ready.json")
    envelope["items"].insert(1, load_fixture("online-invalid.json")["items"][0])
    assert [type(item) for item in _parse(envelope)] == [ReadyWindow, RejectedWindow, ReadyWindow]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "stale_data"),
        ("expectedPoints", 32),
        ("windowStartTimestamp", 9_999_999_999_999),
        ("symbol", "   "),
    ],
)
def test_inconsistent_ready_item_is_rejected(
    load_fixture: LoadFixture, field: str, value: object
) -> None:
    envelope = load_fixture("online-ready.json")
    envelope["items"][0][field] = value
    assert isinstance(_parse(envelope)[0], RejectedWindow)


def test_item_that_is_not_an_object_is_rejected(load_fixture: LoadFixture) -> None:
    envelope = load_fixture("online-ready.json")
    envelope["items"] = [42]
    (item,) = _parse(envelope)
    assert isinstance(item, RejectedWindow)
    assert item.symbol is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeframe", "5m"),
        ("lookbackSteps", 32),
        ("featureColumns", ["open", "rd_value", "high", "low", "close", "volume"]),
        (
            "featureColumns",
            ["rd_value", "open", "high", "low", "close", "volume", "signal_barrier"],
        ),
        ("featureColumns", ["rd_value", "open", "high", "low", "close", "volume", "rating"]),
        ("generatedAt", "2026-10-16T12:00:05"),
        ("items", None),
    ],
)
def test_envelope_header_must_match_contract(
    load_fixture: LoadFixture, field: str, value: object
) -> None:
    envelope = load_fixture("online-ready.json")
    envelope[field] = value
    with pytest.raises(InvalidInputError, match="invalid DS API envelope"):
        parse_envelope(json.dumps(envelope))


def test_malformed_json_is_invalid_input() -> None:
    with pytest.raises(InvalidInputError):
        parse_envelope("{not json")
