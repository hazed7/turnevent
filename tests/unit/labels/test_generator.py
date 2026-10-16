from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from contracts.targets import TurnEvent, TurnState
from dataset.cleaning import CleanDataset, clean_historical
from labels.generator import LabelColumn, LabelConfig, LabelSet, LabelStatus, generate_labels

type MakeClean = Callable[..., CleanDataset]
type RawBars = Callable[..., pd.DataFrame]

CONFIG = LabelConfig()
N = 60
DIP, SPIKE = -5.0, 5.0

uses_fixture = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
random_walks = st.lists(st.floats(-2.0, 2.0), min_size=40, max_size=80).map(
    lambda steps: (100.0 + np.cumsum(steps)).tolist()
)


def _trend(n: int = N) -> list[float]:
    return [100.0 + 0.001 * i for i in range(n)]


def _moved(closes: list[float], moves: dict[int, float]) -> list[float]:
    return [close + moves.get(i, 0.0) for i, close in enumerate(closes)]


def _pinned(closes: list[float], pins: dict[int, float]) -> list[float]:
    return [pins.get(i, close) for i, close in enumerate(closes)]


def _labels(make_clean: MakeClean, closes: list[float], **kwargs: object) -> LabelSet:
    return generate_labels(make_clean(closes, **kwargs), CONFIG)


def _turns(labels: LabelSet) -> dict[int, TurnEvent]:
    events = labels.labelled()[LabelColumn.TURN_EVENT]
    turns = events[events != TurnEvent.NO_TURN]
    return {
        int(i): TurnEvent(int(e))
        for i, e in zip(turns.index.to_numpy(), turns.to_numpy(), strict=True)
    }


def test_deep_dip_is_a_turn_up_and_spike_a_turn_down(make_clean: MakeClean) -> None:
    labels = _labels(make_clean, _moved(_trend(), {25: DIP, 40: SPIKE}))
    assert _turns(labels) == {25: TurnEvent.TURN_UP, 40: TurnEvent.TURN_DOWN}


def test_shallow_pullback_is_no_turn(make_clean: MakeClean) -> None:
    labels = _labels(make_clean, _moved(_trend(), {25: -0.3}))
    assert _turns(labels) == {}
    assert labels.frame.loc[25, LabelColumn.TURN_EVENT] == TurnEvent.NO_TURN


def test_weaker_turn_within_separation_becomes_no_turn(make_clean: MakeClean) -> None:
    labels = _labels(make_clean, _moved(_trend(), {25: DIP, 27: 3.0}))
    assert _turns(labels) == {25: TurnEvent.TURN_UP}


def test_turns_exactly_min_separation_apart_both_survive(make_clean: MakeClean) -> None:
    labels = _labels(make_clean, _moved(_trend(), {25: DIP, 29: DIP}))
    assert _turns(labels) == {25: TurnEvent.TURN_UP, 29: TurnEvent.TURN_UP}


def test_tied_extremum_is_disputed_not_labelled(make_clean: MakeClean) -> None:
    frame = _labels(make_clean, _pinned(_trend(), {25: 90.0, 27: 90.0})).frame
    assert frame.loc[[25, 27], LabelColumn.STATUS].tolist() == [LabelStatus.DISPUTED] * 2
    assert frame.loc[[25, 27], LabelColumn.TURN_EVENT].isna().all()


def test_edges_lack_history_or_confirmation(make_clean: MakeClean) -> None:
    status = _labels(make_clean, _trend()).frame[LabelColumn.STATUS]
    history, future = CONFIG.history_bars, CONFIG.future_bars
    assert (status[:history] == LabelStatus.INSUFFICIENT_HISTORY).all()
    assert (status[N - future :] == LabelStatus.NO_CONFIRMATION).all()
    assert (status[history : N - future] == LabelStatus.LABELLED).all()


def test_bars_whose_span_crosses_a_gap_are_excluded(make_clean: MakeClean) -> None:
    minutes = [*range(1, 31), *range(32, N + 2)]
    status = _labels(make_clean, _trend(), minutes=minutes).frame[LabelColumn.STATUS]
    first, last = 30 - CONFIG.future_bars, 30 + CONFIG.min_separation_bars - 2 + CONFIG.left_bars
    assert (status[first : last + 1] == LabelStatus.GAP).all()
    assert status[first - 1] == LabelStatus.LABELLED
    assert status[last + 1] == LabelStatus.LABELLED


def test_timestamps_record_when_the_label_becomes_known(make_clean: MakeClean) -> None:
    row = _labels(make_clean, _trend()).labelled().iloc[0]
    delay = CONFIG.future_bars * 60_000
    assert row[LabelColumn.EVENT_TIMESTAMP] == row["timestamp"]
    assert row[LabelColumn.LABEL_AVAILABLE_TIMESTAMP] == row["timestamp"] + delay
    assert row[LabelColumn.DETECTION_DELAY_MS] == delay


def test_turn_state_follows_the_latest_turn(make_clean: MakeClean) -> None:
    state = _labels(make_clean, _moved(_trend(), {25: DIP, 40: SPIKE})).frame[
        LabelColumn.TURN_STATE
    ]
    assert (state[CONFIG.history_bars : 25] == TurnState.NEUTRAL).all()
    assert (state[25:40] == TurnState.BUY_HOLD).all()
    assert (state[40 : N - CONFIG.future_bars] == TurnState.SELL_HOLD).all()
    assert state[N - CONFIG.future_bars :].isna().all()


def test_turn_state_resets_after_an_excluded_bar(make_clean: MakeClean) -> None:
    closes = _pinned(_moved(_trend(), {22: DIP}), {33: 90.0, 35: 90.0})
    state = _labels(make_clean, closes).frame[LabelColumn.TURN_STATE]
    assert state[30] == TurnState.BUY_HOLD
    assert state[34] == TurnState.NEUTRAL


def test_symbols_are_labelled_independently(raw_bars: RawBars) -> None:
    dipped = raw_bars(_moved(_trend(), {25: DIP}), symbol="A")
    flat = raw_bars(_trend(), symbol="B")
    clean = clean_historical(pd.concat([dipped, flat], ignore_index=True))
    turns = generate_labels(clean, CONFIG).labelled().groupby("symbol")[LabelColumn.TURN_EVENT]
    assert set(turns.get_group("A")) == {TurnEvent.NO_TURN, TurnEvent.TURN_UP}
    assert set(turns.get_group("B")) == {TurnEvent.NO_TURN}


@pytest.mark.parametrize(("factor", "offset"), [(0.01, 7.0), (3.0, -50.0)])
def test_labels_ignore_price_scale_and_level(
    make_clean: MakeClean, factor: float, offset: float
) -> None:
    closes = (100 + np.cumsum(np.random.default_rng(0).normal(size=N))).tolist()
    expected = _labels(make_clean, closes).frame
    actual = _labels(make_clean, [c * factor + offset for c in closes], band=0.5 * factor).frame
    pd.testing.assert_frame_equal(actual, expected)


@uses_fixture
@given(random_walks)
def test_kept_turns_are_never_closer_than_min_separation(
    make_clean: MakeClean, closes: list[float]
) -> None:
    positions = sorted(_turns(_labels(make_clean, closes)))
    assert all(b - a >= CONFIG.min_separation_bars for a, b in pairwise(positions))


@uses_fixture
@given(random_walks, st.data())
def test_no_label_depends_on_bars_after_its_available_timestamp(
    make_clean: MakeClean, closes: list[float], data: st.DataObject
) -> None:
    cut = data.draw(st.integers(CONFIG.future_bars, len(closes) - 2))
    altered = closes[: cut + 1] + [c * 1.3 + 5.0 for c in closes[cut + 1 :]]
    known = slice(0, cut - CONFIG.future_bars + 1)
    expected = _labels(make_clean, closes).frame.iloc[known]
    pd.testing.assert_frame_equal(_labels(make_clean, altered).frame.iloc[known], expected)


@uses_fixture
@given(random_walks, st.data())
def test_no_label_depends_on_bars_older_than_its_history(
    make_clean: MakeClean, closes: list[float], data: st.DataObject
) -> None:
    cut = data.draw(st.integers(1, len(closes) - CONFIG.history_bars - 1))
    altered = [c * 0.7 + 20.0 for c in closes[:cut]] + closes[cut:]
    known = slice(cut + CONFIG.history_bars, None)
    expected = _labels(make_clean, closes).frame.iloc[known].drop(columns=LabelColumn.TURN_STATE)
    actual = _labels(make_clean, altered).frame.iloc[known].drop(columns=LabelColumn.TURN_STATE)
    pd.testing.assert_frame_equal(actual, expected)
