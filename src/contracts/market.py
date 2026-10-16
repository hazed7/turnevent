from enum import StrEnum, auto
from typing import Final

TIMEFRAME: Final = "1m"
BAR_INTERVAL_MS: Final = 60_000
WINDOW_LENGTH: Final = 64


class Column(StrEnum):
    TIMESTAMP = auto()
    SYMBOL = auto()
    RD_VALUE = auto()
    OPEN = auto()
    HIGH = auto()
    LOW = auto()
    CLOSE = auto()
    VOLUME = auto()
    SIGNAL_BARRIER = auto()


FEATURE_COLUMNS: Final = (
    Column.RD_VALUE,
    Column.OPEN,
    Column.HIGH,
    Column.LOW,
    Column.CLOSE,
    Column.VOLUME,
)
MODEL_INPUT_COLUMNS: Final = (Column.TIMESTAMP, Column.SYMBOL, *FEATURE_COLUMNS)
HISTORICAL_COLUMNS: Final = (*MODEL_INPUT_COLUMNS, Column.SIGNAL_BARRIER)
