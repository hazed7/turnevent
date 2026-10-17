from pathlib import Path

from artifact.bundle import (
    EVENT_MODEL_FILE,
    STATE_MODEL_FILE,
    model_file,
    pack_estimator,
    write_bundle,
)
from artifact.manifest import MetricsSummary, ModelManifest
from core.runtime import RuntimeInfo
from core.serialization import fingerprint
from dataset.manifest import DatasetManifest
from features.schema import build_feature_schema
from labels.definition import EVENT_CLASSES, STATE_CLASSES, build_label_definition
from labels.generator import LABEL_GENERATOR_VERSION, LABEL_SPEC_VERSION
from training.pipeline import Predictor, TrainingConfig, TrainingResult

FEATURE_SCHEMA_FILE = "feature-schema.json"
LABEL_DEFINITION_FILE = "label-definition.json"
SPLIT_MANIFEST_FILE = "split-manifest.json"
TRAINING_CONFIG_FILE = "training-config.json"


def export_bundle(
    result: TrainingResult,
    config: TrainingConfig,
    dataset: DatasetManifest,
    runtime: RuntimeInfo,
    directory: Path,
) -> ModelManifest:
    models = {
        EVENT_MODEL_FILE: pack_estimator(result.event_classifier.estimator),
        STATE_MODEL_FILE: pack_estimator(result.state_classifier.estimator),
    }

    schema = build_feature_schema()
    documents = {
        FEATURE_SCHEMA_FILE: schema.model_dump(mode="json"),
        LABEL_DEFINITION_FILE: build_label_definition(result.labels, dataset.sha256).model_dump(
            mode="json"
        ),
        SPLIT_MANIFEST_FILE: result.split.model_dump(mode="json"),
        TRAINING_CONFIG_FILE: config.model_dump(mode="json"),
    }
    checksums = {name: fingerprint(document) for name, document in documents.items()}

    manifest = ModelManifest(
        model_version=_model_version(models, checksums),
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset.sha256,
        feature_schema_version=schema.builder_version,
        label_schema_version=LABEL_SPEC_VERSION,
        label_generator_version=LABEL_GENERATOR_VERSION,
        prediction_lag_bars=config.prediction_lag_bars,
        normalization_spec=schema.normalization,
        feature_names=result.event_classifier.feature_names,
        event_classes=EVENT_CLASSES,
        state_classes=STATE_CLASSES,
        event_turn_threshold=result.report.selected_turn_threshold,
        event_model=model_file(EVENT_MODEL_FILE, models[EVENT_MODEL_FILE]),
        state_model=model_file(STATE_MODEL_FILE, models[STATE_MODEL_FILE]),
        documents=checksums,
        metrics=_summary(result),
        runtime=runtime,
    )
    write_bundle(directory, manifest, models, documents)
    return manifest


def _model_version(models: dict[str, bytes], document_checksums: dict[str, str]) -> str:
    identity = {
        "models": {name: model_file(name, data).sha256 for name, data in models.items()},
        "documents": document_checksums,
    }
    return f"tm-{fingerprint(identity)[:12]}"


def _summary(result: TrainingResult) -> MetricsSummary:
    report = result.report
    model = report.test_event[Predictor.MODEL]
    return MetricsSummary(
        test_samples=model.samples,
        macro_f1=model.macro_f1,
        balanced_accuracy=model.balanced_accuracy,
        false_turn_rate=model.false_turn_rate,
        detection_rate=model.detection.detection_rate,
        median_delay_bars=model.detection.median_delay_bars,
        majority_macro_f1=report.test_event[Predictor.MAJORITY].macro_f1,
        rd_value_rule_macro_f1=report.test_event[Predictor.RD_VALUE_RULE].macro_f1,
        state_macro_f1=report.test_state.macro_f1,
        event_log_loss=report.test_event_probabilities.log_loss,
        state_log_loss=report.test_state_probabilities.log_loss,
    )
