import pytest
from pydantic import ValidationError

from contracts.market import Column
from dataset.cleaning import CleanDataset
from training.models import Algorithm
from training.pipeline import Predictor, TrainingConfig, TrainingResult, train_and_evaluate


@pytest.fixture(scope="module")
def result(clean: CleanDataset, fast_training: TrainingConfig) -> TrainingResult:
    return train_and_evaluate(clean, fast_training)


def test_reports_model_and_both_baselines_on_the_test_part(result: TrainingResult) -> None:
    assert set(result.report.test_event) == set(Predictor)
    model = result.report.test_event[Predictor.MODEL]
    majority = result.report.test_event[Predictor.MAJORITY]
    assert model.samples == majority.samples == result.split.periods[2].samples
    assert result.report.test_state.samples == model.samples
    assert result.report.labels.labelled > model.samples


def test_rerun_on_the_same_data_and_config_is_identical(
    clean: CleanDataset, result: TrainingResult, fast_training: TrainingConfig
) -> None:
    again = train_and_evaluate(clean, fast_training)
    assert again.report.fingerprint() == result.report.fingerprint()
    assert again.split == result.split


def test_test_period_never_influences_any_choice(
    clean: CleanDataset, result: TrainingResult, fast_training: TrainingConfig
) -> None:
    test_start = result.split.periods[2].start_ms
    frame = clean.frame.copy()
    later = frame[Column.TIMESTAMP] >= test_start
    prices = [Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE]
    frame.loc[later, prices] = frame.loc[later, prices] * 1.5
    frame.loc[later, Column.RD_VALUE] = -frame.loc[later, Column.RD_VALUE]

    altered = train_and_evaluate(CleanDataset(frame=frame, report=clean.report), fast_training)

    assert altered.split.periods[:2] == result.split.periods[:2]
    assert altered.report.selected_turn_threshold == result.report.selected_turn_threshold
    assert altered.report.selected_rd_rule == result.report.selected_rd_rule
    assert altered.report.validation_event == result.report.validation_event
    assert altered.report.selected_algorithm == result.report.selected_algorithm
    assert altered.report.validation_by_algorithm == result.report.validation_by_algorithm


def test_report_records_the_prediction_lag(result: TrainingResult) -> None:
    assert result.report.prediction_lag_bars == 3
    detection = result.report.test_event[Predictor.MODEL].detection
    assert detection.median_delay_bars is None or detection.median_delay_bars >= 3


def test_lag_beyond_the_label_horizon_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not exceed 6"):
        TrainingConfig(prediction_lag_bars=7)


def test_report_scores_the_probabilities_used_for_rating(result: TrainingResult) -> None:
    for metrics in (result.report.test_event_probabilities, result.report.test_state_probabilities):
        assert metrics.log_loss > 0
        assert sum(metrics.mean_probability.values()) == pytest.approx(1.0)


def test_algorithm_with_the_best_validation_score_goes_into_the_bundle(
    result: TrainingResult,
) -> None:
    report = result.report
    assert set(report.validation_by_algorithm) == set(Algorithm)
    assert set(report.test_by_algorithm) == set(Algorithm)
    best = max(report.validation_by_algorithm.values(), key=lambda m: m.macro_f1).macro_f1
    assert report.validation_by_algorithm[report.selected_algorithm].macro_f1 == best
    assert report.validation_event == report.validation_by_algorithm[report.selected_algorithm]
    assert report.test_event[Predictor.MODEL] == report.test_by_algorithm[report.selected_algorithm]


def test_a_single_algorithm_is_always_selected(
    clean: CleanDataset, fast_training: TrainingConfig
) -> None:
    model = fast_training.model.model_copy(
        update={"algorithms": (Algorithm.HIST_GRADIENT_BOOSTING,)}
    )
    config = fast_training.model_copy(update={"model": model})
    report = train_and_evaluate(clean, config).report
    assert report.selected_algorithm is Algorithm.HIST_GRADIENT_BOOSTING
    assert set(report.test_by_algorithm) == {Algorithm.HIST_GRADIENT_BOOSTING}
