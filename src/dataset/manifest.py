from typing import Final

from contracts.market import BAR_INTERVAL_MS, TIMEFRAME
from core.frozen import FrozenModel
from dataset.cleaning import CleanDataset
from dataset.loader import RawDataset, SourceFormat

DATASET_MANIFEST_VERSION: Final = "1.0.0"


class DatasetManifest(FrozenModel):
    manifest_version: str = DATASET_MANIFEST_VERSION
    dataset_version: str
    sha256: str
    source_file: str
    source_format: SourceFormat
    timeframe: str = TIMEFRAME
    bar_interval_ms: int = BAR_INTERVAL_MS
    loaded_columns: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    cleaning_policy_version: str
    rows_raw: int
    rows_clean: int
    n_symbols: int
    period_start_ms: int
    period_end_ms: int


def build_dataset_manifest(raw: RawDataset, clean: CleanDataset) -> DatasetManifest:
    period_start, period_end = clean.period_ms
    return DatasetManifest(
        dataset_version=raw.source.dataset_version,
        sha256=raw.source.sha256,
        source_file=raw.source.file_name,
        source_format=raw.source.source_format,
        loaded_columns=tuple(str(column) for column in raw.frame.columns),
        ignored_columns=raw.source.ignored_columns,
        cleaning_policy_version=clean.report.policy_version,
        rows_raw=len(raw.frame),
        rows_clean=len(clean.frame),
        n_symbols=len(clean.symbols),
        period_start_ms=period_start,
        period_end_ms=period_end,
    )
