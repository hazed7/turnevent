import pytest

from dataset.cleaning import CleanDataset, clean_historical
from dataset.synthetic import SyntheticConfig, generate_synthetic


@pytest.fixture(scope="module")
def clean() -> CleanDataset:
    config = SyntheticConfig(n_symbols=2, bars_per_symbol=300, plant_defects=False)
    raw = generate_synthetic(config).drop(columns="signal_barrier").astype("str")
    return clean_historical(raw)
