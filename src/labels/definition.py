from collections.abc import Mapping
from typing import Final

from contracts.targets import TurnEvent, TurnState
from core.frozen import FrozenModel
from labels.generator import (
    LABEL_GENERATOR_VERSION,
    LABEL_SPEC_VERSION,
    LabelColumn,
    LabelConfig,
    LabelSet,
    LabelStatus,
)


class ClassDefinition(FrozenModel):
    name: str
    code: int
    meaning: str


class LabelSummary(FrozenModel):
    rows: int
    labelled: int
    by_status: Mapping[LabelStatus, int]
    turn_event: Mapping[str, int]
    turn_state: Mapping[str, int]


class LabelExample(FrozenModel):
    turn_event: str
    turn_state: str
    event_timestamp: int
    label_available_timestamp: int
    detection_delay_ms: int


EVENT_CLASSES: Final = (
    ClassDefinition(name="TURN_DOWN", code=-1, meaning="confirmed transition from rise to fall"),
    ClassDefinition(name="NO_TURN", code=0, meaning="no confirmed reversal at this bar"),
    ClassDefinition(name="TURN_UP", code=1, meaning="confirmed transition from fall to rise"),
)
STATE_CLASSES: Final = (
    ClassDefinition(name="SELL_HOLD", code=-1, meaning="latest confirmed turn was TURN_DOWN"),
    ClassDefinition(name="NEUTRAL", code=0, meaning="no confirmed turn since the last reset"),
    ClassDefinition(name="BUY_HOLD", code=1, meaning="latest confirmed turn was TURN_UP"),
)


class LabelDefinition(FrozenModel):
    spec_version: str = LABEL_SPEC_VERSION
    generator_version: str = LABEL_GENERATOR_VERSION
    target: str = "turn_event"
    event_classes: tuple[ClassDefinition, ...] = EVENT_CLASSES
    state_classes: tuple[ClassDefinition, ...] = STATE_CLASSES
    parameters: LabelConfig
    baseline_parameters: LabelConfig = LabelConfig()
    rules: Mapping[str, str]
    dataset_sha256: str
    summary: LabelSummary
    examples: tuple[LabelExample, ...]


def build_label_definition(
    labels: LabelSet, dataset_sha256: str, *, examples_per_class: int = 2
) -> LabelDefinition:
    return LabelDefinition(
        parameters=labels.config,
        rules=describe_rules(labels.config),
        dataset_sha256=dataset_sha256,
        summary=summarize_labels(labels),
        examples=_examples(labels, examples_per_class),
    )


def summarize_labels(labels: LabelSet) -> LabelSummary:
    labelled = labels.labelled()
    statuses = labels.frame[LabelColumn.STATUS].value_counts()
    events = labelled[LabelColumn.TURN_EVENT].value_counts()
    states = labelled[LabelColumn.TURN_STATE].value_counts()
    return LabelSummary(
        rows=len(labels.frame),
        labelled=len(labelled),
        by_status={status: int(statuses.get(status, 0)) for status in LabelStatus},
        turn_event={event.name: int(events.get(event.value, 0)) for event in TurnEvent},
        turn_state={state.name: int(states.get(state.value, 0)) for state in TurnState},
    )


def describe_rules(config: LabelConfig) -> dict[str, str]:
    left, right = config.left_bars, config.right_bars
    separation, window = config.min_separation_bars, config.scale_window_bars
    return {
        "candidate": (
            f"a bar whose close is the strict minimum (TURN_UP) or maximum (TURN_DOWN) "
            f"of the {left}+{right} bars around it"
        ),
        "significance": (
            f"the smaller of the swings into and out of the candidate must reach "
            f"{config.threshold_coefficient} x the trailing {window}-bar average true range"
        ),
        "separation": (
            f"a candidate is kept only if it is stronger than every candidate closer than "
            f"{separation} bars; weaker ones become NO_TURN"
        ),
        "no_turn": "a labelled bar that is not a kept candidate",
        "insufficient_history": f"the first {config.history_bars} bars of each symbol: excluded",
        "no_confirmation": f"the last {config.future_bars} bars of each symbol: excluded",
        "gaps": "a bar whose dependency span crosses a missing 1m bar: excluded, never filled",
        "disputed": "a bar whose close ties the minimum or maximum of its window: excluded",
        "turn_state": (
            "state after the latest kept turn; resets to NEUTRAL at each symbol start "
            "and after every excluded bar"
        ),
        "event_timestamp": "timestamp of the bar being labelled",
        "label_available_timestamp": (
            f"timestamp {config.future_bars} bars later, the last bar the label depends on"
        ),
        "detection_delay": "label_available_timestamp - event_timestamp, in milliseconds",
    }


def _examples(labels: LabelSet, per_class: int) -> tuple[LabelExample, ...]:
    labelled = labels.labelled()
    picked = labelled.groupby(LabelColumn.TURN_EVENT, sort=True).head(per_class)
    return tuple(
        LabelExample(
            turn_event=TurnEvent(int(row[LabelColumn.TURN_EVENT])).name,
            turn_state=TurnState(int(row[LabelColumn.TURN_STATE])).name,
            event_timestamp=int(row[LabelColumn.EVENT_TIMESTAMP]),
            label_available_timestamp=int(row[LabelColumn.LABEL_AVAILABLE_TIMESTAMP]),
            detection_delay_ms=int(row[LabelColumn.DETECTION_DELAY_MS]),
        )
        for _, row in picked.iterrows()
    )
