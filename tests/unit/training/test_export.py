import json
from pathlib import Path

import pytest

from artifact.bundle import load_bundle
from core.runtime import describe_runtime
from core.serialization import fingerprint
from dataset.cleaning import CleanDataset
from dataset.loader import SourceFormat
from dataset.manifest import DatasetManifest
from features.builder import FEATURE_BUILDER_VERSION, FEATURE_NAMES
from labels.generator import LABEL_SPEC_VERSION
from training.export import export_bundle
from training.pipeline import Predictor, TrainingConfig, TrainingResult, train_and_evaluate

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="module")
def result(clean: CleanDataset) -> TrainingResult:
    return train_and_evaluate(clean, TrainingConfig())


@pytest.fixture(scope="module")
def dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_version="ds-000000000000",
        sha256="0" * 64,
        source_file="history.csv",
        source_format=SourceFormat.CSV,
        loaded_columns=(),
        ignored_columns=("signal_barrier",),
        cleaning_policy_version="1.0.0",
        rows_raw=1,
        rows_clean=1,
        n_symbols=1,
        period_start_ms=0,
        period_end_ms=0,
    )


def _export(result: TrainingResult, dataset: DatasetManifest, directory: Path) -> None:
    export_bundle(result, TrainingConfig(), dataset, describe_runtime(PROJECT_ROOT), directory)


def test_manifest_carries_every_field_the_sow_requires(
    result: TrainingResult, dataset: DatasetManifest, tmp_path: Path
) -> None:
    _export(result, dataset, tmp_path)
    manifest = load_bundle(tmp_path).manifest

    assert manifest.artifact_kind == "turnevent-classifier-bundle"
    assert manifest.model_version.startswith("tm-")
    assert manifest.dataset_version == dataset.dataset_version
    assert manifest.feature_schema_version == FEATURE_BUILDER_VERSION
    assert manifest.label_schema_version == LABEL_SPEC_VERSION
    assert manifest.window_length == 64
    assert manifest.prediction_lag_bars == 3
    assert "nothing is fitted" in manifest.normalization_spec
    assert manifest.feature_names == FEATURE_NAMES
    assert [c.code for c in manifest.event_classes] == [-1, 0, 1]
    assert [c.code for c in manifest.state_classes] == [-1, 0, 1]
    assert manifest.metrics.macro_f1 == result.report.test_event[Predictor.MODEL].macro_f1


def test_bundle_documents_match_their_checksums(
    result: TrainingResult, dataset: DatasetManifest, tmp_path: Path
) -> None:
    _export(result, dataset, tmp_path)
    manifest = load_bundle(tmp_path).manifest
    assert set(manifest.documents) == {
        "feature-schema.json",
        "label-definition.json",
        "split-manifest.json",
        "training-config.json",
    }
    for name, checksum in manifest.documents.items():
        assert fingerprint(json.loads((tmp_path / name).read_text())) == checksum


def test_exporting_the_same_result_twice_is_byte_identical(
    result: TrainingResult, dataset: DatasetManifest, tmp_path: Path
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _export(result, dataset, first)
    _export(result, dataset, second)
    for file in first.iterdir():
        assert file.read_bytes() == (second / file.name).read_bytes(), file.name
