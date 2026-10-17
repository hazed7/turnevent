import pytest
from pydantic import ValidationError

from contracts.market import Column
from dataset.cleaning import CleanDataset
from training.pipeline import Predictor, TrainingConfig, TrainingResult, train_and_evaluate


@pytest.fixture(scope="module")
def result(clean: CleanDataset) -> TrainingResult:
    return train_and_evaluate(clean, TrainingConfig())


def test_reports_model_and_both_baselines_on_the_test_part(result: TrainingResult) -> None:
    assert set(result.report.test_event) == set(Predictor)
    model = result.report.test_event[Predictor.MODEL]
    majority = result.report.test_event[Predictor.MAJORITY]
    assert model.samples == majority.samples == result.split.periods[2].samples
    assert result.report.test_state.samples == model.samples
    assert result.report.labels.labelled > model.samples


def test_rerun_on_the_same_data_and_config_is_identical(
    clean: CleanDataset, result: TrainingResult
) -> None:
    again = train_and_evaluate(clean, TrainingConfig())
    assert again.report.fingerprint() == result.report.fingerprint()
    assert again.split == result.split


def test_test_period_never_influences_any_choice(
    clean: CleanDataset, result: TrainingResult
) -> None:
    test_start = result.split.periods[2].start_ms
    frame = clean.frame.copy()
    later = frame[Column.TIMESTAMP] >= test_start
    prices = [Column.OPEN, Column.HIGH, Column.LOW, Column.CLOSE]
    frame.loc[later, prices] = frame.loc[later, prices] * 1.5
    frame.loc[later, Column.RD_VALUE] = -frame.loc[later, Column.RD_VALUE]

    altered = train_and_evaluate(CleanDataset(frame=frame, report=clean.report), TrainingConfig())

    assert altered.split.periods[:2] == result.split.periods[:2]
    assert altered.report.selected_turn_threshold == result.report.selected_turn_threshold
    assert altered.report.selected_rd_rule_min_change == result.report.selected_rd_rule_min_change
    assert altered.report.validation_event == result.report.validation_event


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
