from collections.abc import Callable
from pathlib import Path

from contracts.market import MODEL_INPUT_COLUMNS
from core.serialization import sha256_file
from dataset.cleaning import clean_historical
from dataset.loader import load_historical
from dataset.manifest import build_dataset_manifest


def test_manifest_identifies_the_dataset(write_csv: Callable[..., Path]) -> None:
    path = write_csv(
        "60000,A,0.5,10,11,9,10.5,100,1",
        "120000,B,0.5,10,11,9,10.5,100,1",
        "bad,B,0.5,10,11,9,10.5,100,1",
    )
    raw = load_historical(path)

    manifest = build_dataset_manifest(raw, clean_historical(raw.frame))

    assert manifest.sha256 == sha256_file(path)
    assert manifest.dataset_version == f"ds-{manifest.sha256[:12]}"
    assert manifest.loaded_columns == MODEL_INPUT_COLUMNS
    assert manifest.ignored_columns == ("signal_barrier",)
    assert (manifest.rows_raw, manifest.rows_clean, manifest.n_symbols) == (3, 2, 2)
    assert (manifest.period_start_ms, manifest.period_end_ms) == (60_000, 120_000)
    assert (manifest.timeframe, manifest.bar_interval_ms) == ("1m", 60_000)
