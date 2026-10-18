from collections.abc import Callable

import pytest

from contracts.targets import TurnEvent, TurnState
from inference.results import LastBar, Prediction, WindowResult, WindowStatus

type MakeResult = Callable[..., WindowResult]


@pytest.fixture
def make_result() -> MakeResult:
    def make(
        turn_event: TurnEvent = TurnEvent.TURN_UP,
        *,
        symbol: str = "SYM001",
        window_end: int = 1_760_000_000_000,
        status: WindowStatus = WindowStatus.PREDICTED,
        state_probabilities: tuple[float, float, float] = (0.1, 0.2, 0.7),
    ) -> WindowResult:
        predicted = status is WindowStatus.PREDICTED
        down, neutral, up = state_probabilities
        return WindowResult(
            symbol=symbol,
            window_end_timestamp=window_end,
            status=status,
            reason=None if predicted else "stale_data",
            prediction=Prediction(
                turn_event=turn_event,
                event_probabilities={"TURN_DOWN": 0.1, "NO_TURN": 0.2, "TURN_UP": 0.7},
                turn_state=TurnState.BUY_HOLD,
                state_probabilities={"SELL_HOLD": down, "NEUTRAL": neutral, "BUY_HOLD": up},
            )
            if predicted
            else None,
            last_bar=LastBar(close=101.5, rd_value=0.3, volume=1200.0) if predicted else None,
            model_version="tm-test",
            feature_schema_version="1.1.0",
        )

    return make
