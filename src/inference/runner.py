from typing import Final

import numpy as np
import pandas as pd

from artifact.bundle import LoadedBundle
from artifact.classifier import CLASS_CODES
from contracts.market import FEATURE_COLUMNS, Column
from contracts.online import PendingWindow, ReadyWindow, RejectedWindow, parse_envelope
from contracts.targets import TurnEvent, TurnState
from core.errors import InvalidInputError
from features.builder import build_features
from features.windows import windows_from_ready
from inference.results import InferenceBatch, LastBar, Prediction, WindowResult, WindowStatus

_UNUSABLE_WINDOW: Final = "window has a non-positive price, negative volume or non-finite value"
_PROBABILITY_TOLERANCE: Final = 1e-6

type Item = ReadyWindow | PendingWindow | RejectedWindow


def run_inference(payload: str | bytes, bundle: LoadedBundle) -> InferenceBatch:
    try:
        envelope = parse_envelope(payload)
    except InvalidInputError as exc:
        return InferenceBatch(
            generated_at=None,
            model_version=bundle.manifest.model_version,
            envelope_error=str(exc),
            results=(),
        )
    return InferenceBatch(
        generated_at=envelope.generated_at,
        model_version=bundle.manifest.model_version,
        envelope_error=None,
        results=_process(envelope.items, bundle),
    )


def _process(items: tuple[Item, ...], bundle: LoadedBundle) -> tuple[WindowResult, ...]:
    results: dict[int, WindowResult] = {}
    ready: dict[int, ReadyWindow] = {}
    features: list[pd.DataFrame] = []

    for position, item in enumerate(items):
        if isinstance(item, ReadyWindow):
            try:
                features.append(_features(item))
                ready[position] = item
            except InvalidInputError as exc:
                results[position] = _without_prediction(
                    item, WindowStatus.INVALID_INPUT, str(exc), bundle
                )
        elif isinstance(item, PendingWindow):
            reason = item.reason.value if item.reason else item.state.value.lower()
            results[position] = _without_prediction(item, WindowStatus.NOT_READY, reason, bundle)
        else:
            results[position] = _without_prediction(
                item, WindowStatus.INVALID_INPUT, item.detail, bundle
            )

    if ready:
        results |= _predict(ready, pd.concat(features, ignore_index=True), bundle)
    return tuple(results[position] for position in range(len(items)))


def _features(item: ReadyWindow) -> pd.DataFrame:
    batch = windows_from_ready([item])
    if not batch.usable().all():
        raise InvalidInputError(_UNUSABLE_WINDOW)
    return build_features(batch)


def _predict(
    ready: dict[int, ReadyWindow], features: pd.DataFrame, bundle: LoadedBundle
) -> dict[int, WindowResult]:
    try:
        event_probabilities = bundle.event_classifier.probabilities(features)
        state_probabilities = bundle.state_classifier.probabilities(features)
        _check_probabilities(event_probabilities)
        _check_probabilities(state_probabilities)

        events = bundle.event_classifier.predict(features)
        states = bundle.state_classifier.predict(features)
    except Exception as exc:  # noqa: BLE001
        reason = f"model failed: {type(exc).__name__}: {exc}"
        return {
            position: _without_prediction(item, WindowStatus.MODEL_ERROR, reason, bundle)
            for position, item in ready.items()
        }

    return {
        position: _with_prediction(
            item,
            Prediction(
                turn_event=TurnEvent(int(events[row])),
                event_probabilities=_named(event_probabilities.iloc[row], TurnEvent),
                turn_state=TurnState(int(states[row])),
                state_probabilities=_named(state_probabilities.iloc[row], TurnState),
            ),
            bundle,
        )
        for row, (position, item) in enumerate(ready.items())
    }


def _check_probabilities(probabilities: pd.DataFrame) -> None:
    values = probabilities.to_numpy()
    valid = np.isfinite(values).all() and (values >= 0).all()

    if not valid or not np.allclose(values.sum(axis=1), 1.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("model returned invalid probabilities")


def _named(row: pd.Series, classes: type[TurnEvent] | type[TurnState]) -> dict[str, float]:
    return {classes(code).name: float(row[code]) for code in CLASS_CODES}


def _with_prediction(
    item: ReadyWindow,
    prediction: Prediction,
    bundle: LoadedBundle,
) -> WindowResult:
    last = item.features[-1]
    return WindowResult(
        symbol=item.symbol,
        window_end_timestamp=item.window_end_timestamp,
        status=WindowStatus.PREDICTED,
        reason=None,
        prediction=prediction,
        last_bar=LastBar(
            close=last[FEATURE_COLUMNS.index(Column.CLOSE)],
            rd_value=last[FEATURE_COLUMNS.index(Column.RD_VALUE)],
            volume=last[FEATURE_COLUMNS.index(Column.VOLUME)],
        ),
        model_version=bundle.manifest.model_version,
        feature_schema_version=bundle.manifest.feature_schema_version,
    )


def _without_prediction(
    item: Item,
    status: WindowStatus,
    reason: str,
    bundle: LoadedBundle,
) -> WindowResult:
    return WindowResult(
        symbol=item.symbol,
        window_end_timestamp=None
        if isinstance(item, RejectedWindow)
        else item.window_end_timestamp,
        status=status,
        reason=reason,
        prediction=None,
        last_bar=None,
        model_version=bundle.manifest.model_version,
        feature_schema_version=bundle.manifest.feature_schema_version,
    )
