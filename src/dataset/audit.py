from typing import Final

import pandas as pd

from contracts.market import BAR_INTERVAL_MS, WINDOW_LENGTH, Column
from core.frozen import FrozenModel
from core.runtime import RuntimeInfo
from dataset.cleaning import CleanDataset, CleaningReport
from dataset.loader import RawDataset, SourceInfo

AUDIT_SCHEMA_VERSION: Final = "1.0.0"


class Periodicity(FrozenModel):
    expected_interval_ms: int
    median_step_ms: float | None
    regular_step_share: float | None
    gaps: int
    missing_bars: int
    irregular_steps: int


class SymbolCoverage(FrozenModel):
    symbol: str
    rows: int
    first_timestamp_ms: int
    last_timestamp_ms: int
    gaps: int
    missing_bars: int
    windows: int


class DataAudit(FrozenModel):
    schema_version: str = AUDIT_SCHEMA_VERSION
    source: SourceInfo
    loaded_columns: tuple[str, ...]
    rows_raw: int
    rows_clean: int
    period_start_ms: int
    period_end_ms: int
    cleaning: CleaningReport
    periodicity: Periodicity
    window_length: int
    available_windows: int
    symbols: tuple[SymbolCoverage, ...]
    runtime: RuntimeInfo


def build_data_audit(
    raw: RawDataset,
    clean: CleanDataset,
    runtime: RuntimeInfo,
    *,
    interval_ms: int = BAR_INTERVAL_MS,
) -> DataAudit:
    steps = clean.frame.groupby(Column.SYMBOL)[Column.TIMESTAMP].diff()
    coverage = _symbol_coverage(clean.frame, steps, interval_ms)
    period_start, period_end = clean.period_ms

    return DataAudit(
        source=raw.source,
        loaded_columns=tuple(str(column) for column in raw.frame.columns),
        rows_raw=len(raw.frame),
        rows_clean=len(clean.frame),
        period_start_ms=period_start,
        period_end_ms=period_end,
        cleaning=clean.report,
        periodicity=_periodicity(steps.dropna(), interval_ms),
        window_length=WINDOW_LENGTH,
        available_windows=sum(symbol.windows for symbol in coverage),
        symbols=coverage,
        runtime=runtime,
    )


def _periodicity(steps: pd.Series, interval_ms: int) -> Periodicity:
    has_steps = not steps.empty
    return Periodicity(
        expected_interval_ms=interval_ms,
        median_step_ms=float(steps.median()) if has_steps else None,
        regular_step_share=float((steps == interval_ms).mean()) if has_steps else None,
        gaps=int((steps > interval_ms).sum()),
        missing_bars=int(_missing_bars(steps, interval_ms).sum()),
        irregular_steps=int((steps % interval_ms != 0).sum()),
    )


def _symbol_coverage(
    frame: pd.DataFrame, steps: pd.Series, interval_ms: int
) -> tuple[SymbolCoverage, ...]:
    per_row = pd.DataFrame(
        {
            Column.SYMBOL: frame[Column.SYMBOL],
            Column.TIMESTAMP: frame[Column.TIMESTAMP],
            "gap": steps > interval_ms,
            "missing": _missing_bars(steps, interval_ms),
        },
    )
    grouped = per_row.groupby(Column.SYMBOL, sort=True).agg(
        rows=(Column.TIMESTAMP, "size"),
        first=(Column.TIMESTAMP, "min"),
        last=(Column.TIMESTAMP, "max"),
        gaps=("gap", "sum"),
        missing=("missing", "sum"),
    )

    return tuple(
        SymbolCoverage(
            symbol=str(symbol),
            rows=int(rows),
            first_timestamp_ms=int(first),
            last_timestamp_ms=int(last),
            gaps=int(gaps),
            missing_bars=int(missing),
            windows=max(int(rows) - WINDOW_LENGTH + 1, 0),
        )
        for symbol, rows, first, last, gaps, missing in grouped.itertuples(name=None)
    )


def _missing_bars(steps: pd.Series, interval_ms: int) -> pd.Series:
    return (steps // interval_ms - 1).clip(lower=0).fillna(0).astype(int)
