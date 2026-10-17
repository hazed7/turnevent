from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Self

from pydantic import NonNegativeInt, model_validator

from artifact.classifier import TurnClassifier
from core.frozen import FrozenModel
from dataset.cleaning import CleanDataset
from features.builder import build_offline_features
from labels.definition import LabelSummary, summarize_labels
from labels.generator import LabelConfig, LabelSet, generate_labels
from training.baselines import MajorityBaseline, RdValueCrossBaseline
from training.metrics import (
    EVENT_NAMES,
    STATE_NAMES,
    ClassificationMetrics,
    EvaluationConfig,
    EventMetrics,
    ProbabilityMetrics,
    classification_metrics,
    event_metrics,
    probability_metrics,
)
from training.models import ModelConfig, fit_event_classifier, fit_state_classifier
from training.samples import assemble_samples
from training.split import Part, SplitConfig, SplitManifest, time_split


class Predictor(StrEnum):
    MAJORITY = auto()
    RD_VALUE_RULE = auto()
    MODEL = auto()


class TrainingConfig(FrozenModel):
    labels: LabelConfig = LabelConfig()
    prediction_lag_bars: NonNegativeInt = 3
    split: SplitConfig = SplitConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def _lag_within_label_horizon(self) -> Self:
        if self.prediction_lag_bars > self.labels.future_bars:
            raise ValueError(f"prediction_lag_bars must not exceed {self.labels.future_bars}")
        return self


class EvaluationReport(FrozenModel):
    labels: LabelSummary
    prediction_lag_bars: int
    selected_turn_threshold: float
    selected_rd_rule_min_change: float
    validation_event: EventMetrics
    test_event: Mapping[Predictor, EventMetrics]
    test_state: ClassificationMetrics
    test_event_probabilities: ProbabilityMetrics
    test_state_probabilities: ProbabilityMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    event_classifier: TurnClassifier
    state_classifier: TurnClassifier
    labels: LabelSet
    split: SplitManifest
    report: EvaluationReport


def train_and_evaluate(clean: CleanDataset, config: TrainingConfig) -> TrainingResult:
    labels = generate_labels(clean, config.labels)
    samples = assemble_samples(
        clean,
        build_offline_features(clean),
        labels,
        prediction_lag_bars=config.prediction_lag_bars,
    )
    parts, split = time_split(samples, config.split)
    train, validation, test = parts[Part.TRAIN], parts[Part.VALIDATION], parts[Part.TEST]

    event_classifier = fit_event_classifier(train, validation, config.model)
    state_classifier = fit_state_classifier(train, config.model)
    majority = MajorityBaseline.fit(train.turn_event)
    rd_rule = RdValueCrossBaseline.select(validation.features, validation.turn_event)

    test_predictions = {
        Predictor.MAJORITY: majority.predict(test.features),
        Predictor.RD_VALUE_RULE: rd_rule.predict(test.features),
        Predictor.MODEL: event_classifier.predict(test.features),
    }
    report = EvaluationReport(
        labels=summarize_labels(labels),
        prediction_lag_bars=config.prediction_lag_bars,
        selected_turn_threshold=event_classifier.turn_threshold or 0.0,
        selected_rd_rule_min_change=rd_rule.min_change,
        validation_event=event_metrics(
            validation.turn_event,
            event_classifier.predict(validation.features),
            validation.keys,
            config.evaluation,
        ),
        test_event={
            name: event_metrics(test.turn_event, predicted, test.keys, config.evaluation)
            for name, predicted in test_predictions.items()
        },
        test_state=classification_metrics(
            test.turn_state, state_classifier.predict(test.features), STATE_NAMES
        ),
        test_event_probabilities=probability_metrics(
            test.turn_event, event_classifier.probabilities(test.features), EVENT_NAMES
        ),
        test_state_probabilities=probability_metrics(
            test.turn_state, state_classifier.probabilities(test.features), STATE_NAMES
        ),
    )
    return TrainingResult(
        event_classifier=event_classifier,
        state_classifier=state_classifier,
        labels=labels,
        split=split,
        report=report,
    )
