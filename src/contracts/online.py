from enum import StrEnum, auto
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    Discriminator,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import to_json

from contracts.market import FEATURE_COLUMNS, TIMEFRAME, WINDOW_LENGTH
from core.errors import InvalidInputError
from core.frozen import FrozenModel

type Symbol = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
type FeatureRow = Annotated[
    tuple[FiniteFloat, ...], Field(min_length=len(FEATURE_COLUMNS), max_length=len(FEATURE_COLUMNS))
]
type FeatureMatrix = Annotated[
    tuple[FeatureRow, ...], Field(min_length=WINDOW_LENGTH, max_length=WINDOW_LENGTH)
]


class WindowState(StrEnum):
    READY = "READY"
    WARMUP = "WARMUP"


class NotReadyReason(StrEnum):
    NOT_ENOUGH_POINTS = auto()
    STALE_DATA = auto()
    MISSING_JOINED_ROWS = auto()


class _WindowItem(FrozenModel):
    symbol: Symbol
    points_collected: Annotated[NonNegativeInt, Field(alias="pointsCollected")]
    expected_points: Annotated[NonNegativeInt, Field(alias="expectedPoints")]
    window_start_timestamp: Annotated[NonNegativeInt, Field(alias="windowStartTimestamp")]
    window_end_timestamp: Annotated[NonNegativeInt, Field(alias="windowEndTimestamp")]

    @model_validator(mode="after")
    def _matches_window_geometry(self) -> Self:
        if self.expected_points != WINDOW_LENGTH:
            raise ValueError(f"expectedPoints must be {WINDOW_LENGTH}")
        if self.window_start_timestamp > self.window_end_timestamp:
            raise ValueError("windowStartTimestamp is after windowEndTimestamp")
        return self


class ReadyWindow(_WindowItem):
    state: Literal[WindowState.READY]
    reason: None
    features: FeatureMatrix

    @model_validator(mode="after")
    def _window_is_complete(self) -> Self:
        if self.points_collected != WINDOW_LENGTH:
            raise ValueError(f"READY window must have {WINDOW_LENGTH} points")
        return self


class PendingWindow(_WindowItem):
    state: Literal[WindowState.WARMUP]
    reason: NotReadyReason | None
    features: None = None


class RejectedWindow(FrozenModel):
    symbol: str | None
    detail: str


type WindowItem = Annotated[ReadyWindow | PendingWindow, Discriminator("state")]


class _Envelope(FrozenModel):
    timeframe: str
    lookback_steps: Annotated[int, Field(alias="lookbackSteps")]
    feature_columns: Annotated[tuple[str, ...], Field(alias="featureColumns")]
    generated_at: Annotated[AwareDatetime, Field(alias="generatedAt")]
    items: tuple[JsonValue, ...]

    @model_validator(mode="after")
    def _matches_contract(self) -> Self:
        expected = (TIMEFRAME, WINDOW_LENGTH, FEATURE_COLUMNS)
        actual = (self.timeframe, self.lookback_steps, self.feature_columns)

        if actual != expected:
            raise ValueError(f"envelope heaer {actual} does not match contract {expected}")
        return self


class OnlineBatch(FrozenModel):
    generated_at: AwareDatetime
    items: tuple[ReadyWindow | PendingWindow | RejectedWindow, ...]


_WINDOW_ITEM: TypeAdapter[WindowItem] = TypeAdapter(WindowItem)


def parse_envelope(payload: str | bytes) -> OnlineBatch:
    try:
        envelope = _Envelope.model_validate_json(payload)
    except ValidationError as exc:
        raise InvalidInputError(f"invalid DS API envelope:\n{exc}") from exc

    return OnlineBatch(
        generated_at=envelope.generated_at,
        items=tuple(_parse_item(item) for item in envelope.items),
    )


def _parse_item(item: JsonValue) -> ReadyWindow | PendingWindow | RejectedWindow:
    try:
        return _WINDOW_ITEM.validate_json(to_json(item))
    except ValidationError as exc:
        symbol = item.get("symbol") if isinstance(item, dict) else None
        return RejectedWindow(symbol=symbol if isinstance(symbol, str) else None, detail=str(exc))
