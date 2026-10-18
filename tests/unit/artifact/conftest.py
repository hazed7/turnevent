from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from core.runtime import describe_runtime
from dataset.cleaning import clean_historical
from dataset.loader import load_historical
from dataset.manifest import build_dataset_manifest
from dataset.synthetic import SyntheticConfig, write_synthetic_csv
from features.builder import FEATURE_NAMES, build_offline_features
from training.export import export_bundle
from training.pipeline import TrainingConfig, TrainingResult, train_and_evaluate

PROJECT_ROOT = Path(__file__).parents[3]


@dataclass(frozen=True)
class Trained:
    result: TrainingResult
    directory: Path
    features: pd.DataFrame


@pytest.fixture(scope="package")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    source = tmp_path_factory.mktemp("data") / "history.csv"
    write_synthetic_csv(SyntheticConfig(n_symbols=2, bars_per_symbol=3_000), source)
    raw = load_historical(source)
    clean = clean_historical(raw.frame)
    result = train_and_evaluate(clean, TrainingConfig())
    directory = tmp_path_factory.mktemp("bundle")
    dataset = build_dataset_manifest(raw, clean)
    export_bundle(result, TrainingConfig(), dataset, describe_runtime(PROJECT_ROOT), directory)
    features = build_offline_features(clean)[list(FEATURE_NAMES)].head(500)
    return Trained(result=result, directory=directory, features=features)


@pytest.fixture(scope="package")
def trained_result(trained: Trained) -> TrainingResult:
    return trained.result


@pytest.fixture(scope="package")
def sample_features(trained: Trained) -> pd.DataFrame:
    return trained.features


@pytest.fixture
def bundle_dir(trained: Trained, tmp_path: Path) -> Path:
    copy = tmp_path / "bundle"
    copy.mkdir()
    for file in trained.directory.iterdir():
        (copy / file.name).write_bytes(file.read_bytes())
    return copy
