import pytest

from dataset.cleaning import CleanDataset, clean_historical
from dataset.synthetic import SyntheticConfig, generate_synthetic
from features.builder import build_offline_features
from labels.generator import LabelConfig, generate_labels
from training.samples import Samples, assemble_samples

LAG = 3


@pytest.fixture(scope="package")
def clean() -> CleanDataset:
    config = SyntheticConfig(n_symbols=2, bars_per_symbol=3_000, plant_defects=False)
    raw = generate_synthetic(config).drop(columns="signal_barrier").astype("str")
    return clean_historical(raw)


@pytest.fixture(scope="package")
def samples(clean: CleanDataset) -> Samples:
    labels = generate_labels(clean, LabelConfig())
    return assemble_samples(clean, build_offline_features(clean), labels, prediction_lag_bars=LAG)
