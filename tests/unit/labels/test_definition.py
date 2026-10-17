import json
from collections.abc import Callable

from contracts.targets import TurnEvent, TurnState
from core.serialization import to_pretty_json
from dataset.cleaning import CleanDataset
from labels.definition import LabelDefinition, build_label_definition, summarize_labels
from labels.generator import LabelConfig, LabelStatus, generate_labels

type MakeClean = Callable[..., CleanDataset]


def _closes() -> list[float]:
    closes = [100.0 + 0.001 * i for i in range(60)]
    closes[25] -= 5.0
    closes[40] += 5.0
    return closes


def test_summary_counts_every_bar_once(make_clean: MakeClean) -> None:
    labels = generate_labels(make_clean(_closes()), LabelConfig())
    summary = summarize_labels(labels)
    assert summary.rows == 60
    assert sum(summary.by_status.values()) == summary.rows
    assert summary.by_status[LabelStatus.LABELLED] == summary.labelled
    assert sum(summary.turn_event.values()) == summary.labelled
    assert sum(summary.turn_state.values()) == summary.labelled
    assert (summary.turn_event["TURN_UP"], summary.turn_event["TURN_DOWN"]) == (1, 1)


def test_definition_documents_classes_rules_and_parameters(make_clean: MakeClean) -> None:
    config = LabelConfig(threshold_coefficient=0.8)
    definition = build_label_definition(generate_labels(make_clean(_closes()), config), "ab" * 32)

    assert [(c.name, c.code) for c in definition.event_classes] == [
        (event.name, event.value) for event in TurnEvent
    ]
    assert [(c.name, c.code) for c in definition.state_classes] == [
        (state.name, state.value) for state in TurnState
    ]
    assert definition.parameters == config
    assert definition.baseline_parameters == LabelConfig()
    assert "0.8 x the trailing 14-bar average true range" in definition.rules["significance"]
    assert set(definition.rules) >= {"gaps", "disputed", "insufficient_history", "no_confirmation"}


def test_examples_cover_each_class_without_symbols(make_clean: MakeClean) -> None:
    labels = generate_labels(make_clean(_closes(), symbol="SECRET_TICKER"), LabelConfig())
    definition = build_label_definition(labels, "ab" * 32, examples_per_class=1)
    assert sorted(example.turn_event for example in definition.examples) == [
        "NO_TURN",
        "TURN_DOWN",
        "TURN_UP",
    ]
    assert "SECRET_TICKER" not in to_pretty_json(definition.model_dump(mode="json"))


def test_definition_round_trips_through_json(make_clean: MakeClean) -> None:
    labels = generate_labels(make_clean(_closes()), LabelConfig())
    definition = build_label_definition(labels, "ab" * 32)
    text = to_pretty_json(definition.model_dump(mode="json"))
    assert LabelDefinition.model_validate(json.loads(text), strict=False) == definition
