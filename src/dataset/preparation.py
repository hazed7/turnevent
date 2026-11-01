"""Validate frozen CSV exports and stream a deterministic model-input Parquet."""

import csv
import io
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

from contracts.market import BAR_INTERVAL_MS, HISTORICAL_COLUMNS, MODEL_INPUT_COLUMNS
from core.errors import DataContractError
from core.serialization import sha256_bytes, sha256_file, to_pretty_json

CHECKSUMS: Final = Path("manifests/2026-10-sha256sums.txt")
DAY_MS: Final = 86_400_000
ENTRY: Final = re.compile(
    r"([0-9a-f]{64})  (raw/marketscreener/(\d{4}-\d{2}-\d{2})/([A-Za-z0-9_-]+)\.csv)"
)
_TYPES: Final[dict[str, pa.DataType]] = {
    "timestamp": pa.int64(),
    "symbol": pa.string(),
    **{str(column): pa.float64() for column in MODEL_INPUT_COLUMNS[2:]},
}
SCHEMA: Final = pa.schema(_TYPES)


@dataclass(frozen=True)
class InputFile:
    relative: str
    day: date
    symbol: str
    sha256: str


@dataclass(frozen=True)
class Selection:
    start_date: date | None = None
    end_date: date | None = None
    symbols: tuple[str, ...] = ()

    def includes(self, item: InputFile) -> bool:
        return (
            (self.start_date is None or item.day >= self.start_date)
            and (self.end_date is None or item.day <= self.end_date)
            and (not self.symbols or item.symbol in self.symbols)
        )


DEFAULT_SELECTION: Final = Selection()


class FileRecord(TypedDict):
    path: str
    sha256: str
    bytes: int
    rows: int
    first_timestamp: int
    last_timestamp: int
    internal_missing_minutes: int


def _inventory(source: Path, checksums: Path) -> list[InputFile]:
    entries = []
    for number, line in enumerate(checksums.read_text(encoding="utf-8").splitlines(), 1):
        match = ENTRY.fullmatch(line)
        if match is None:
            raise DataContractError(f"invalid checksum entry at line {number}")
        digest, relative, day, symbol = match.groups()
        entries.append(InputFile(relative, date.fromisoformat(day), symbol, digest))
    paths = [item.relative for item in entries]
    if not paths or len(paths) != len(set(paths)):
        raise DataContractError("checksum inventory is empty or contains duplicate paths")
    for item in entries:
        path = source / item.relative
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise DataContractError(f"missing or unsafe source file: {item.relative}")
    # Only inspect months represented in this release; other releases may coexist.
    actual = {
        path.relative_to(source).as_posix()
        for month in {item.day.isoformat()[:7] for item in entries}
        for path in (source / "raw/marketscreener").glob(f"{month}-*/*.csv")
    }
    if actual != set(paths):
        raise DataContractError("CSV inventory differs from checksum list (extra/missing files)")
    return sorted(entries, key=lambda item: (item.symbol, item.day))


def _select(entries: Iterable[InputFile], selection: Selection) -> list[InputFile]:
    if (
        selection.start_date is not None
        and selection.end_date is not None
        and selection.start_date > selection.end_date
    ):
        raise DataContractError("start date must be on or before end date")
    selected = [item for item in entries if selection.includes(item)]
    missing = set(selection.symbols) - {item.symbol for item in selected}
    if not selected or missing:
        raise DataContractError(f"selection has no files or unknown symbols: {sorted(missing)}")
    return selected


def _row(row: list[str], item: InputFile, day_start: int) -> tuple[int, str, *tuple[float, ...]]:
    if len(row) != len(HISTORICAL_COLUMNS):
        raise DataContractError(f"ragged CSV row: {item.relative}")
    if not row[0].isdigit():
        raise DataContractError(f"invalid integer timestamp: {item.relative}")
    timestamp = int(row[0])
    if timestamp % BAR_INTERVAL_MS or not day_start <= timestamp < day_start + DAY_MS:
        raise DataContractError(f"timestamp outside UTC day/minute grid: {item.relative}")
    if row[1] != item.symbol:
        raise DataContractError(f"symbol differs from filename: {item.relative}")
    try:
        values = tuple(float(value) for value in row[2:8])
    except ValueError as exc:
        raise DataContractError(f"invalid numeric input: {item.relative}") from exc
    if not all(math.isfinite(value) for value in values):
        raise DataContractError(f"non-finite input: {item.relative}")
    _, opening, high, low, close, volume = values
    if min(opening, high, low, close) <= 0 or volume < 0 or not low <= min(opening, close):
        raise DataContractError(f"invalid price/volume/OHLC: {item.relative}")
    if high < max(opening, close) or low > high:
        raise DataContractError(f"invalid OHLC range: {item.relative}")
    # signal_barrier is intentionally never parsed or used in validation/selection.
    return timestamp, item.symbol, *values


def _read(source: Path, item: InputFile) -> tuple[pa.Table, FileRecord]:
    data = (source / item.relative).read_bytes()
    if sha256_bytes(data) != item.sha256:
        raise DataContractError(f"SHA256 mismatch: {item.relative}")
    day_start = int(datetime.combine(item.day, datetime.min.time(), tzinfo=UTC).timestamp()) * 1000
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True)
        if tuple(next(reader, ())) != HISTORICAL_COLUMNS:
            raise DataContractError(f"unexpected CSV schema: {item.relative}")
        rows = sorted((_row(row, item, day_start) for row in reader), key=lambda row: row[0])
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataContractError(f"malformed CSV: {item.relative}: {exc}") from exc
    if not rows:
        raise DataContractError(f"empty CSV: {item.relative}")
    times = [row[0] for row in rows]
    if len(set(times)) != len(times):
        raise DataContractError(f"duplicate/conflicting timestamp: {item.relative}")
    gaps = sum((right - left) // BAR_INTERVAL_MS - 1 for left, right in pairwise(times))
    table = pa.Table.from_arrays(
        [
            pa.array([row[index] for row in rows], type=field.type)
            for index, field in enumerate(SCHEMA)
        ],
        schema=SCHEMA,
    )
    return table, {
        "path": item.relative,
        "sha256": item.sha256,
        "bytes": len(data),
        "rows": len(rows),
        "first_timestamp": times[0],
        "last_timestamp": times[-1],
        "internal_missing_minutes": gaps,
    }


def _write_selected(
    source: Path, selected: list[InputFile], target: Path
) -> tuple[list[FileRecord], int]:
    records = []
    previous: dict[str, int] = {}
    boundary_missing = 0
    with pq.ParquetWriter(target, SCHEMA, compression="zstd") as writer:
        for item in selected:
            table, record = _read(source, item)
            first = int(table.column("timestamp")[0].as_py())
            last = int(table.column("timestamp")[-1].as_py())
            if item.symbol in previous:
                delta = first - previous[item.symbol]
                if delta <= 0:
                    raise DataContractError(f"overlapping files: {item.relative}")
                boundary_missing += delta // BAR_INTERVAL_MS - 1
            previous[item.symbol] = last
            writer.write_table(table)
            records.append(record)
    return records, boundary_missing


def prepare_dataset(
    source: Path,
    output: Path,
    selection: Selection = DEFAULT_SELECTION,
    checksums: Path = CHECKSUMS,
) -> dict[str, object]:
    """Verify inventory, validate selected files, publish Parquet + provenance without overwrite."""
    if output.suffix != ".parquet":
        raise DataContractError("output must have .parquet extension")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if any(path.exists() or path.is_symlink() for path in (output, manifest_path)):
        raise DataContractError(
            "output or preparation manifest already exists; choose a new output"
        )
    checksum_path = source / checksums
    entries = _inventory(source, checksum_path)
    selected = _select(entries, selection)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        target = Path(temporary) / "dataset.parquet"
        records, boundary_missing = _write_selected(source, selected, target)
        counts = Counter(item.symbol for item in selected)
        manifest: dict[str, object] = {
            "format_version": 1,
            "preparation_version": "1",
            "pyarrow_version": pa.__version__,
            "source_checksums_sha256": sha256_file(checksum_path),
            "validation_scope": "entire inventory presence; selected CSV hashes and contents",
            "inventory_files": len(entries),
            "selected_files": len(selected),
            "selection": {
                "start_date": str(selection.start_date) if selection.start_date else None,
                "end_date": str(selection.end_date) if selection.end_date else None,
                "symbols": sorted(set(selection.symbols)),
            },
            "available_selected_dates": sorted({item.day.isoformat() for item in selected}),
            "files_per_symbol": dict(sorted(counts.items())),
            "columns": list(MODEL_INPUT_COLUMNS),
            "excluded_columns": ["signal_barrier"],
            "sort_order": ["symbol", "timestamp"],
            "rows": sum(record["rows"] for record in records),
            "internal_missing_minutes": sum(
                record["internal_missing_minutes"] for record in records
            ),
            "between_file_missing_minutes": boundary_missing,
            "gap_policy": "preserve timestamps; never fill; enforce window continuity downstream",
            "conflict_policy": "reject duplicate timestamps; never silently choose a row",
            "output_sha256": sha256_file(target),
            "output_bytes": target.stat().st_size,
            "inputs": records,
        }
        temporary_manifest = Path(temporary) / "manifest.json"
        temporary_manifest.write_text(to_pretty_json(manifest), encoding="utf-8")
        # Hard links publish exclusively: never replace a concurrently created destination.
        os.link(target, output)
        try:
            os.link(temporary_manifest, manifest_path)
        except OSError:
            output.unlink()
            raise
    return manifest
