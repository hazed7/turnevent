from collections.abc import Callable
from pathlib import Path

from contracts.market import MODEL_INPUT_COLUMNS
from core.runtime import RuntimeInfo
from dataset.audit import DataAudit, build_data_audit
from dataset.cleaning import clean_historical
from dataset.loader import load_historical

type WriteCsv = Callable[..., Path]

RUNTIME = RuntimeInfo(python="3.13.0", lockfile_sha256="0" * 64, packages={"pandas": "3.0.0"})


def _bars(symbol: str, minutes: list[int]) -> list[str]:
    return [f"{minute * 60_000},{symbol},0.5,10,11,9,10.5,100,1" for minute in minutes]


def _audit(write_csv: WriteCsv, *rows: str) -> DataAudit:
    raw = load_historical(write_csv(*rows))
    return build_data_audit(raw, clean_historical(raw.frame), RUNTIME)


def test_counts_windows_gaps_and_missing_bars_per_symbol(write_csv: WriteCsv) -> None:
    gapped = [*range(1, 41), *range(44, 74)]
    audit = _audit(write_csv, *_bars("A", gapped), *_bars("B", list(range(1, 11))))

    a, b = audit.symbols
    assert (a.symbol, a.rows, a.gaps, a.missing_bars, a.windows) == ("A", 70, 1, 3, 7)
    assert (b.symbol, b.rows, b.gaps, b.missing_bars, b.windows) == ("B", 10, 0, 0, 0)
    assert a.first_timestamp_ms == 60_000
    assert a.last_timestamp_ms == 73 * 60_000
    assert audit.available_windows == 7
    assert audit.periodicity.gaps == 1
    assert audit.periodicity.missing_bars == 3
    assert audit.periodicity.median_step_ms == 60_000


def test_irregular_steps_are_reported(write_csv: WriteCsv) -> None:
    audit = _audit(write_csv, "60000,A,0.5,10,11,9,10.5,100,1", "150000,A,0.5,10,11,9,10.5,100,1")
    assert audit.periodicity.irregular_steps == 1
    assert audit.periodicity.regular_step_share == 0.0


def test_single_row_per_symbol_has_no_step_statistics(write_csv: WriteCsv) -> None:
    audit = _audit(write_csv, *_bars("A", [1]), *_bars("B", [1]))
    assert audit.periodicity.median_step_ms is None
    assert audit.periodicity.regular_step_share is None


def test_records_what_was_loaded_not_what_was_promised(write_csv: WriteCsv) -> None:
    audit = _audit(write_csv, *_bars("A", [1, 2]))
    assert audit.source.signal_barrier_present
    assert audit.loaded_columns == MODEL_INPUT_COLUMNS
    assert audit.source.column_types["signal_barrier"] == "text"
    assert audit.runtime == RUNTIME
    assert audit.fingerprint() == _audit(write_csv, *_bars("A", [1, 2])).fingerprint()
