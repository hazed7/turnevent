import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from contracts.market import FEATURE_COLUMNS, WINDOW_LENGTH, Column
from contracts.online import ReadyWindow, parse_envelope
from core.errors import InvalidInputError
from dataset.cleaning import CleanDataset
from features.builder import (
    FEATURE_BUILDER_VERSION,
    FEATURE_NAMES,
    build_features,
    build_offline_features,
)
from features.windows import WindowBatch, windows_from_ready

type LoadFixture = Callable[[str], Any]

PRICE_FEATURES = [name for name in FEATURE_NAMES if not name.startswith(("rd_value", "volume"))]


def _ready_windows(envelope: str) -> list[ReadyWindow]:
    items = parse_envelope(envelope).items
    assert all(isinstance(item, ReadyWindow) for item in items)
    return [item for item in items if isinstance(item, ReadyWindow)]


def _ready(load_fixture: LoadFixture, name: str = "online-ready.json") -> list[ReadyWindow]:
    return _ready_windows(json.dumps(load_fixture(name)))


def _envelope_for(rows: pd.DataFrame) -> str:
    return json.dumps(
        {
            "timeframe": "1m",
            "lookbackSteps": WINDOW_LENGTH,
            "featureColumns": list(FEATURE_COLUMNS),
            "generatedAt": "2026-01-01T00:00:00Z",
            "items": [
                {
                    "symbol": str(rows[Column.SYMBOL].iloc[0]),
                    "state": "READY",
                    "reason": None,
                    "pointsCollected": WINDOW_LENGTH,
                    "expectedPoints": WINDOW_LENGTH,
                    "windowStartTimestamp": int(rows[Column.TIMESTAMP].iloc[0]),
                    "windowEndTimestamp": int(rows[Column.TIMESTAMP].iloc[-1]),
                    "features": rows[list(FEATURE_COLUMNS)].to_numpy().tolist(),
                },
            ],
        },
    )


def test_offline_and_online_paths_produce_identical_features(clean: CleanDataset) -> None:
    offline = build_offline_features(clean).set_index([Column.SYMBOL, Column.TIMESTAMP])
    for symbol, rows in clean.frame.groupby(Column.SYMBOL):
        for end in (WINDOW_LENGTH, 150, len(rows)):
            window = rows.iloc[end - WINDOW_LENGTH : end]
            online = build_features(windows_from_ready(_ready_windows(_envelope_for(window))))
            expected = offline.loc[[(symbol, window[Column.TIMESTAMP].iloc[-1])]]
            np.testing.assert_array_equal(online.to_numpy(), expected.to_numpy())


def test_offline_features_cover_every_full_window(clean: CleanDataset) -> None:
    features = build_offline_features(clean)
    per_symbol = clean.frame.groupby(Column.SYMBOL).size() - WINDOW_LENGTH + 1
    assert len(features) == per_symbol.sum()
    assert list(features.columns) == [Column.SYMBOL, Column.TIMESTAMP, *FEATURE_NAMES]
    assert np.isfinite(features[list(FEATURE_NAMES)].to_numpy()).all()


def test_a_feature_row_depends_only_on_its_own_window(clean: CleanDataset) -> None:
    cut = 200
    prices = [Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE]
    frame = clean.frame.copy()
    frame.loc[cut + 1 :, prices] *= 1.7
    frame.loc[: cut - WINDOW_LENGTH, prices] *= 0.3

    baseline = build_offline_features(clean)
    altered = build_offline_features(CleanDataset(frame=frame, report=clean.report))

    row = cut - (WINDOW_LENGTH - 1)
    pd.testing.assert_series_equal(altered.iloc[row], baseline.iloc[row])
    assert not altered.iloc[row - 1].equals(baseline.iloc[row - 1])
    assert not altered.iloc[row + 1].equals(baseline.iloc[row + 1])


def test_price_features_ignore_the_price_level(load_fixture: LoadFixture) -> None:
    batch = windows_from_ready(_ready(load_fixture))
    scaled = batch.matrices.copy()
    scaled[:, :, 1:5] *= 37.0
    rescaled = WindowBatch(scaled, batch.start_ms, batch.end_ms)
    np.testing.assert_allclose(
        build_features(rescaled)[PRICE_FEATURES], build_features(batch)[PRICE_FEATURES], atol=1e-12
    )


def test_flat_window_has_well_defined_features() -> None:
    matrices = np.tile([0.0, 10.0, 10.0, 10.0, 10.0, 0.0], (1, WINDOW_LENGTH, 1))
    span = (WINDOW_LENGTH - 1) * 60_000
    flat = WindowBatch(matrices, np.array([0]), np.array([span]))
    row = build_features(flat).iloc[0]
    assert row["return_63"] == row["volatility_20"] == row["rd_value_zscore_64"] == 0.0
    assert row["range_position_20"] == 0.5
    assert row["missing_bars"] == 0.0


def test_gaps_inside_a_window_are_counted_not_filled(load_fixture: LoadFixture) -> None:
    gapped = _ready(load_fixture)[1]
    assert build_features(windows_from_ready([gapped])).loc[0, "missing_bars"] == 7.0


def test_unusable_window_is_rejected(load_fixture: LoadFixture) -> None:
    batch = windows_from_ready(_ready(load_fixture))
    batch.matrices[0, 0, FEATURE_COLUMNS.index(Column.CLOSE)] = -1.0
    with pytest.raises(InvalidInputError, match="positive prices"):
        build_features(batch)


def test_overflowing_window_is_rejected() -> None:
    matrices = np.tile([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], (1, WINDOW_LENGTH, 1))
    matrices[0, -1, 1:5] = 1e308
    matrices[0, -2, 1:5] = 1e-308
    batch = WindowBatch(matrices, np.array([0]), np.array([0]))
    with pytest.raises(InvalidInputError, match="non-finite"):
        build_features(batch)


def test_dataset_without_a_full_window_is_rejected(clean: CleanDataset) -> None:
    short = CleanDataset(frame=clean.frame.iloc[:10], report=clean.report)
    with pytest.raises(InvalidInputError, match="no symbol has 64 clean rows"):
        build_offline_features(short)


def test_values_change_only_with_a_version_bump(load_fixture: LoadFixture) -> None:
    golden = load_fixture("features-golden.json")
    assert golden["builder_version"] == FEATURE_BUILDER_VERSION, (
        "builder version changed: regenerate fixtures/features-golden.json"
    )
    actual = build_features(windows_from_ready(_ready(load_fixture, golden["source"])))
    for row, expected in zip(actual.to_dict("records"), golden["windows"], strict=True):
        assert row == pytest.approx(expected["features"], rel=1e-12, abs=1e-15), (
            "feature values changed: bump FEATURE_BUILDER_VERSION and regenerate the golden file"
        )
