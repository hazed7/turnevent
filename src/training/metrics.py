from collections.abc import Mapping
from itertools import pairwise
from typing import Final

import numpy as np
import pandas as pd
from pydantic import NonNegativeInt, PositiveInt
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support

from artifact.classifier import CLASS_CODES, Codes
from contracts.market import BAR_INTERVAL_MS, Column
from contracts.targets import TurnEvent, TurnState
from core.frozen import FrozenModel
from training.samples import SampleColumn

EVENT_NAMES: Final = {event.value: event.name for event in TurnEvent}
STATE_NAMES: Final = {state.value: state.name for state in TurnState}


class EvaluationConfig(FrozenModel):
    delay_tolerance_bars: NonNegativeInt = 6
    time_segments: PositiveInt = 4


class ClassMetrics(FrozenModel):
    precision: float
    recall: float
    f1: float
    support: int


class ClassificationMetrics(FrozenModel):
    samples: int
    macro_f1: float
    balanced_accuracy: float
    per_class: Mapping[str, ClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]
    true_share: Mapping[str, float]
    predicted_share: Mapping[str, float]


class ProbabilityMetrics(FrozenModel):
    log_loss: float
    brier_score: float
    mean_probability: Mapping[str, float]
    true_share: Mapping[str, float]


class DetectionDelay(FrozenModel):
    true_turns: int
    detected: int
    detection_rate: float | None
    median_delay_bars: float | None
    mean_delay_bars: float | None


class SegmentMetrics(FrozenModel):
    start_ms: int
    end_ms: int
    samples: int
    macro_f1: float


class EventMetrics(ClassificationMetrics):
    false_turn_rate: float | None
    detection: DetectionDelay
    segments: tuple[SegmentMetrics, ...]


def classification_metrics(
    truth: Codes, predicted: Codes, names: Mapping[int, str]
) -> ClassificationMetrics:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=CLASS_CODES, zero_division=0
    )
    return ClassificationMetrics(
        samples=len(truth),
        macro_f1=float(np.mean(f1)),
        balanced_accuracy=float(recall[support > 0].mean()),
        per_class={
            names[code]: ClassMetrics(
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, code in enumerate(CLASS_CODES)
        },
        confusion_matrix=tuple(
            tuple(int(cell) for cell in row)
            for row in confusion_matrix(truth, predicted, labels=CLASS_CODES)
        ),
        true_share=_shares(truth, names),
        predicted_share=_shares(predicted, names),
    )


def event_metrics(
    truth: Codes, predicted: Codes, keys: pd.DataFrame, config: EvaluationConfig
) -> EventMetrics:
    base = classification_metrics(truth, predicted, EVENT_NAMES)
    predicted_turns = predicted != TurnEvent.NO_TURN
    return EventMetrics(
        **dict(base),
        false_turn_rate=(
            float((truth[predicted_turns] != predicted[predicted_turns]).mean())
            if predicted_turns.any()
            else None
        ),
        detection=_detection_delay(truth, predicted, keys, config.delay_tolerance_bars),
        segments=_segments(truth, predicted, keys, config.time_segments),
    )


def probability_metrics(
    truth: Codes, probabilities: pd.DataFrame, names: Mapping[int, str]
) -> ProbabilityMetrics:
    values = probabilities[list(CLASS_CODES)].to_numpy()
    one_hot = (truth[:, None] == np.asarray(CLASS_CODES)[None, :]).astype(np.float64)
    return ProbabilityMetrics(
        log_loss=float(log_loss(truth, values, labels=CLASS_CODES)),
        brier_score=float(((values - one_hot) ** 2).sum(axis=1).mean()),
        mean_probability={
            names[code]: float(values[:, i].mean()) for i, code in enumerate(CLASS_CODES)
        },
        true_share=_shares(truth, names),
    )


def _shares(codes: Codes, names: Mapping[int, str]) -> dict[str, float]:
    return {names[code]: float((codes == code).mean()) for code in CLASS_CODES}


def _detection_delay(
    truth: Codes, predicted: Codes, keys: pd.DataFrame, tolerance_bars: int
) -> DetectionDelay:
    frame = keys[[Column.SYMBOL, Column.TIMESTAMP, SampleColumn.EVENT_TIMESTAMP]].assign(
        truth=truth, predicted=predicted
    )
    true_turns = frame.loc[frame["truth"] != TurnEvent.NO_TURN].rename(
        columns={SampleColumn.EVENT_TIMESTAMP: "at", "truth": "direction"}
    )
    calls = frame.loc[frame["predicted"] != TurnEvent.NO_TURN].rename(
        columns={Column.TIMESTAMP: "at", "predicted": "direction"}
    )
    matched = pd.merge_asof(
        true_turns[[Column.SYMBOL, "at", "direction"]].sort_values("at"),
        calls[[Column.SYMBOL, "at", "direction"]].assign(called_at=calls["at"]).sort_values("at"),
        on="at",
        by=[Column.SYMBOL, "direction"],
        direction="forward",
        tolerance=tolerance_bars * BAR_INTERVAL_MS,
    )
    delays = ((matched["called_at"] - matched["at"]) / BAR_INTERVAL_MS).dropna()
    return DetectionDelay(
        true_turns=len(true_turns),
        detected=len(delays),
        detection_rate=len(delays) / len(true_turns) if len(true_turns) else None,
        median_delay_bars=float(delays.median()) if len(delays) else None,
        mean_delay_bars=float(delays.mean()) if len(delays) else None,
    )


def _segments(
    truth: Codes, predicted: Codes, keys: pd.DataFrame, count: int
) -> tuple[SegmentMetrics, ...]:
    timestamps = keys[Column.TIMESTAMP].to_numpy()
    edges = np.linspace(timestamps.min(), timestamps.max() + 1, count + 1).astype(np.int64)
    segments = []
    for start, end in pairwise(edges):
        inside = (timestamps >= start) & (timestamps < end)
        if inside.any():
            segments.append(
                SegmentMetrics(
                    start_ms=int(start),
                    end_ms=int(end),
                    samples=int(inside.sum()),
                    macro_f1=classification_metrics(
                        truth[inside], predicted[inside], EVENT_NAMES
                    ).macro_f1,
                ),
            )
    return tuple(segments)
