from contracts.targets import TurnEvent, TurnState


def test_class_codes_match_sow_mapping() -> None:
    assert {member.name: member.value for member in TurnEvent} == {
        "TURN_DOWN": -1,
        "NO_TURN": 0,
        "TURN_UP": 1,
    }
    assert {member.name: member.value for member in TurnState} == {
        "SELL_HOLD": -1,
        "NEUTRAL": 0,
        "BUY_HOLD": 1,
    }
