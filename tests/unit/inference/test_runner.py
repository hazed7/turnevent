import dataclasses
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from artifact.bundle import LoadedBundle
from artifact.classifier import TurnClassifier
from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH, Column
from dataset.cleaning import CleanDataset
from features.builder import FEATURE_NAMES, build_offline_features
from inference.results import WindowResult, WindowStatus
from inference.runner import run_inference

type LoadFixture = Callable[[str], Any]

PROBABILITY_KEYS = {"TURN_DOWN", "NO_TURN", "TURN_UP"}


def _run(bundle: LoadedBundle, envelope: dict[str, Any]) -> tuple[WindowResult, ...]:
    batch = run_inference(json.dumps(envelope), bundle)
    assert batch.envelope_error is None
    return batch.results


def test_ready_windows_get_full_predictions(
    bundle: LoadedBundle, load_fixture: LoadFixture
) -> None:
    envelope = load_fixture("online-ready.json")
    results = _run(bundle, envelope)
    assert [r.status for r in results] == [WindowStatus.PREDICTED] * 2
    for result, item in zip(results, envelope["items"], strict=True):
        assert result.prediction is not None
        assert result.last_bar is not None
        assert set(result.prediction.event_probabilities) == PROBABILITY_KEYS
        assert sum(result.prediction.event_probabilities.values()) == pytest.approx(1.0)
        assert sum(result.prediction.state_probabilities.values()) == pytest.approx(1.0)
        assert result.window_end_timestamp == item["windowEndTimestamp"]
        assert result.last_bar.close == item["features"][-1][FEATURE_COLUMNS.index(Column.CLOSE)]
        assert result.model_version == bundle.manifest.model_version
        assert result.feature_schema_version == bundle.manifest.feature_schema_version


def test_warmup_windows_are_skipped_with_their_reason(
    bundle: LoadedBundle, load_fixture: LoadFixture
) -> None:
    results = _run(bundle, load_fixture("online-warmup.json"))
    assert {r.status for r in results} == {WindowStatus.NOT_READY}
    assert [r.reason for r in results] == [
        "not_enough_points",
        "stale_data",
        "missing_joined_rows",
        "warmup",
    ]
    assert all(r.prediction is None for r in results)


def test_invalid_windows_never_reach_the_model(
    bundle: LoadedBundle, load_fixture: LoadFixture
) -> None:
    results = _run(bundle, load_fixture("online-invalid.json"))
    assert {r.status for r in results} == {WindowStatus.INVALID_INPUT}
    assert [r.symbol for r in results] == [f"BAD00{n}" for n in range(1, 8)]
    assert all(r.prediction is None and r.reason for r in results)


def test_mixed_envelope_keeps_item_order(bundle: LoadedBundle, load_fixture: LoadFixture) -> None:
    envelope = load_fixture("online-ready.json")
    envelope["items"] = [
        load_fixture("online-warmup.json")["items"][0],
        envelope["items"][0],
        load_fixture("online-invalid.json")["items"][0],
        envelope["items"][1],
    ]
    assert [r.status for r in _run(bundle, envelope)] == [
        WindowStatus.NOT_READY,
        WindowStatus.PREDICTED,
        WindowStatus.INVALID_INPUT,
        WindowStatus.PREDICTED,
    ]


def test_ready_window_with_impossible_prices_is_invalid(
    bundle: LoadedBundle, load_fixture: LoadFixture
) -> None:
    envelope = load_fixture("online-ready.json")
    envelope["items"][0]["features"][10][FEATURE_COLUMNS.index(Column.LOW)] = -1.0
    first, second = _run(bundle, envelope)
    assert first.status is WindowStatus.INVALID_INPUT
    assert first.reason is not None
    assert "non-positive price" in first.reason
    assert second.status is WindowStatus.PREDICTED


def test_broken_envelope_is_reported_without_results(bundle: LoadedBundle) -> None:
    batch = run_inference("{not json", bundle)
    assert batch.results == ()
    assert batch.envelope_error is not None
    assert batch.model_version == bundle.manifest.model_version


class _Broken:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, _features: pd.DataFrame) -> np.ndarray[Any, Any]:
        raise RuntimeError("boom")


class _Nonsense:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray[Any, Any]:
        return np.full((len(features), 3), 0.5)


@pytest.mark.parametrize(("estimator", "message"), [(_Broken(), "boom"), (_Nonsense(), "invalid")])
def test_model_failures_become_per_window_errors(
    bundle: LoadedBundle, load_fixture: LoadFixture, estimator: object, message: str
) -> None:
    broken = dataclasses.replace(
        bundle,
        event_classifier=TurnClassifier(FEATURE_NAMES, estimator, turn_threshold=0.5),  # type: ignore[arg-type]
    )
    results = _run(broken, load_fixture("online-ready.json"))
    assert {r.status for r in results} == {WindowStatus.MODEL_ERROR}
    assert all(r.reason is not None and message in r.reason for r in results)


def test_online_probabilities_match_offline_ones(
    bundle: LoadedBundle, trained_clean: CleanDataset
) -> None:
    rows = trained_clean.frame.loc[trained_clean.frame[Column.SYMBOL] == "SYM000"].iloc[
        -WINDOW_LENGTH:
    ]
    envelope = {
        "timeframe": "1m",
        "lookbackSteps": WINDOW_LENGTH,
        "featureColumns": list(FEATURE_COLUMNS),
        "generatedAt": "2026-01-01T00:00:00Z",
        "items": [
            {
                "symbol": "SYM000",
                "state": "READY",
                "reason": None,
                "pointsCollected": WINDOW_LENGTH,
                "expectedPoints": WINDOW_LENGTH,
                "windowStartTimestamp": int(rows[Column.TIMESTAMP].iloc[0]),
                "windowEndTimestamp": int(rows[Column.TIMESTAMP].iloc[-1]),
                "features": rows[list(FEATURE_COLUMNS)].to_numpy().tolist(),
            },
        ],
    }
    (online,) = _run(bundle, envelope)
    offline = build_offline_features(trained_clean)
    last = offline.loc[offline[Column.SYMBOL] == "SYM000", list(FEATURE_NAMES)].tail(1)
    expected = bundle.event_classifier.probabilities(last.reset_index(drop=True)).iloc[0]
    assert online.prediction is not None
    assert list(online.prediction.event_probabilities.values()) == expected.tolist()


def test_result_invariants_are_enforced() -> None:
    with pytest.raises(ValidationError, match="exactly when status is PREDICTED"):
        WindowResult(
            symbol="A",
            window_end_timestamp=1,
            status=WindowStatus.PREDICTED,
            reason=None,
            prediction=None,
            last_bar=None,
            model_version="tm-x",
            feature_schema_version="1.1.0",
        )
    with pytest.raises(ValidationError, match="reason is given"):
        WindowResult(
            symbol="A",
            window_end_timestamp=1,
            status=WindowStatus.NOT_READY,
            reason=None,
            prediction=None,
            last_bar=None,
            model_version="tm-x",
            feature_schema_version="1.1.0",
        )
