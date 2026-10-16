from enum import StrEnum, auto
from typing import Annotated

from pydantic import Field, FiniteFloat, NonNegativeInt

from core.frozen import FrozenModel

type Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


class Signal(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Source(StrEnum):
    ML = auto()
    SHADOW = auto()


class SignalPayload(FrozenModel):
    symbol: Annotated[str, Field(min_length=1)]
    timestamp: NonNegativeInt
    signal: Signal
    price: Annotated[FiniteFloat, Field(gt=0.0)]
    rating: Probability
    source: Source
