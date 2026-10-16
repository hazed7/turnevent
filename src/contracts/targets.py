from enum import IntEnum


class TurnEvent(IntEnum):
    TURN_DOWN = -1
    NO_TURN = 0
    TURN_UP = 1


class TurnState(IntEnum):
    SELL_HOLD = -1
    NEUTRAL = 0
    BUY_HOLD = 1
