import json
import math

import pytest
from pydantic import ValidationError

from contracts.payload import Signal, SignalPayload, Source

VALID = {
    "symbol": "SYM001",
    "timestamp": 1_760_007_000_000,
    "signal": "BUY",
    "price": 101.25,
    "rating": 0.71,
    "source": "ml",
}


def test_valid_payload_round_trips_through_json() -> None:
    payload = SignalPayload.model_validate_json(json.dumps(VALID))
    assert payload.signal is Signal.BUY
    assert payload.source is Source.ML
    assert json.loads(payload.model_dump_json()) == VALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("signal", "HOLD"),
        ("signal", "NO_TURN"),
        ("source", "live"),
        ("price", 0.0),
        ("price", -1.0),
        ("rating", 1.01),
        ("rating", -0.01),
        ("symbol", ""),
        ("timestamp", -1),
    ],
)
def test_out_of_contract_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SignalPayload.model_validate_json(json.dumps({**VALID, field: value}))


def test_non_finite_rating_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SignalPayload.model_validate({**VALID, "rating": math.nan}, strict=False)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        SignalPayload.model_validate_json(json.dumps({**VALID, "signal_barrier": 1}))
