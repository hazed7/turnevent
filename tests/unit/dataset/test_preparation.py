import hashlib
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from contracts.market import HISTORICAL_COLUMNS, MODEL_INPUT_COLUMNS
from core.errors import DataContractError
from dataset.loader import load_historical
from dataset.preparation import CHECKSUMS, Selection, prepare_dataset

START = 1792368000000  # 2026-10-19 00:00 UTC


def _write(source: Path, symbol: str, day: str, rows: list[str]) -> Path:
    path = source / f"raw/marketscreener/{day}/{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(HISTORICAL_COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _row(timestamp: int, symbol: str = "RLC", *, barrier: str = "ignored") -> str:
    return f"{timestamp},{symbol},-1.2,0.7445,0.7445,0.7437,0.7439,2451,{barrier}"


def _checksums(source: Path) -> None:
    path = source / CHECKSUMS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{hashlib.sha256(item.read_bytes()).hexdigest()}  "
            f"{item.relative_to(source).as_posix()}\n"
            for item in sorted(source.glob("raw/marketscreener/*/*.csv"))
        ),
        encoding="utf-8",
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write(source, "RLC", "2026-10-19", [_row(START + 120000), _row(START)])
    _write(source, "RLC", "2026-10-20", [_row(START + 86400000)])
    _write(source, "ABC", "2026-10-19", [_row(START, "ABC")])
    _checksums(source)
    return source


def test_streams_sorted_parquet_with_provenance_and_preserves_gaps(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared.parquet"
    result = prepare_dataset(source, output)
    table = pq.read_table(output)
    assert table.column_names == list(MODEL_INPUT_COLUMNS)
    assert table.column("symbol").to_pylist() == ["ABC", "RLC", "RLC", "RLC"]
    assert table.column("timestamp").to_pylist() == [START, START, START + 120000, START + 86400000]
    assert result["rows"] == 4
    assert result["internal_missing_minutes"] == 1
    assert result["between_file_missing_minutes"] == 1437
    assert result["excluded_columns"] == ["signal_barrier"]
    assert output.with_suffix(".parquet.manifest.json").is_file()
    loaded = load_historical(output)
    assert len(loaded.frame) == 4
    assert not loaded.source.signal_barrier_present
    # Same input produces byte-identical data and manifest, irrespective of target name.
    other = tmp_path / "other.parquet"
    assert prepare_dataset(source, other) == result
    assert output.read_bytes() == other.read_bytes()


def test_selects_dates_and_symbols(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = prepare_dataset(
        source,
        tmp_path / "prepared.parquet",
        Selection(date(2026, 10, 20), date(2026, 10, 20), ("RLC",)),
    )
    assert result["selected_files"] == 1
    assert result["inventory_files"] == 3
    assert result["rows"] == 1


@pytest.mark.parametrize(
    "selection",
    [
        Selection(symbols=("UNKNOWN",)),
        Selection(date(2026, 10, 20), date(2026, 10, 19)),
        Selection(date(2026, 10, 21), date(2026, 10, 22)),
    ],
)
def test_rejects_empty_or_invalid_selection(tmp_path: Path, selection: Selection) -> None:
    with pytest.raises(DataContractError):
        prepare_dataset(_source(tmp_path), tmp_path / "out.parquet", selection)


@pytest.mark.parametrize("kind", ["missing", "modified", "extra"])
def test_rejects_inventory_or_hash_mismatch(tmp_path: Path, kind: str) -> None:
    source = _source(tmp_path)
    item = source / "raw/marketscreener/2026-10-19/RLC.csv"
    if kind == "missing":
        item.unlink()
    elif kind == "modified":
        item.write_text(item.read_text() + "\n")
    else:
        _write(source, "EXTRA", "2026-10-19", [_row(START, "EXTRA")])
    output = tmp_path / "out.parquet"
    with pytest.raises(DataContractError):
        prepare_dataset(source, output)
    assert not output.exists()
    assert not output.with_suffix(".parquet.manifest.json").exists()
    assert not list(tmp_path.glob(".prepare-*"))


@pytest.mark.parametrize(
    "rows",
    [
        [_row(START), _row(START)],
        [_row(START + 1)],
        [
            _row(START).replace(
                str(START), str(START).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
            )
        ],
        ["²" + _row(START)],
        [_row(START - 60000)],
        [_row(START, "WRONG")],
        [_row(START).replace("-1.2", "nan")],
        [_row(START).replace("-1.2", "invalid")],
        [_row(START).replace("0.7439", "10")],
        [_row(START) + ",extra"],
        [],
    ],
)
def test_rejects_bad_source_content_even_with_matching_checksum(
    tmp_path: Path, rows: list[str]
) -> None:
    source = tmp_path / "source"
    _write(source, "RLC", "2026-10-19", rows)
    _checksums(source)
    with pytest.raises(DataContractError):
        prepare_dataset(source, tmp_path / "out.parquet")


@pytest.mark.parametrize("existing", ["out.parquet", "out.parquet.manifest.json"])
def test_never_overwrites_existing_outputs(tmp_path: Path, existing: str) -> None:
    source = _source(tmp_path)
    destination = tmp_path / existing
    destination.write_bytes(b"keep")
    with pytest.raises(DataContractError, match="already exists"):
        prepare_dataset(source, tmp_path / "out.parquet")
    assert destination.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "edit", [lambda text: text.replace("  raw/", " *raw/"), lambda text: f"\n{text}\n"]
)
def test_accepts_binary_mode_checksums_and_blank_lines(
    tmp_path: Path, edit: Callable[[str], str]
) -> None:
    source = _source(tmp_path)
    checksums = source / CHECKSUMS
    checksums.write_text(edit(checksums.read_text()))
    assert prepare_dataset(source, tmp_path / "out.parquet")["inventory_files"] == 3


def test_skips_blank_csv_lines(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source, "RLC", "2026-10-19", [_row(START), "", _row(START + 60000), ""])
    _checksums(source)
    assert prepare_dataset(source, tmp_path / "out.parquet")["rows"] == 2


def test_rejects_path_traversal_in_inventory(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / CHECKSUMS).write_text("a" * 64 + "  ../../outside.csv\n")
    with pytest.raises(DataContractError, match="invalid checksum entry"):
        prepare_dataset(source, tmp_path / "out.parquet")


@pytest.mark.parametrize("existing", ["out.parquet", "out.parquet.manifest.json"])
def test_preserves_dangling_output_symlinks(tmp_path: Path, existing: str) -> None:
    source = _source(tmp_path)
    path = tmp_path / existing
    path.symlink_to("missing-target")
    with pytest.raises(DataContractError, match="already exists"):
        prepare_dataset(source, tmp_path / "out.parquet")
    assert path.is_symlink()


def test_failed_manifest_publication_rolls_back_only_own_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "out.parquet"
    manifest = output.with_suffix(".parquet.manifest.json")
    link = os.link

    def racing_link(source: Path, target: Path) -> None:
        if target == manifest:
            manifest.write_text("concurrent writer")
        link(source, target)

    monkeypatch.setattr("core.files.os.link", racing_link)
    with pytest.raises(FileExistsError):
        prepare_dataset(source, output)
    assert not output.exists()
    assert manifest.read_text() == "concurrent writer"
    assert not list(tmp_path.glob(".prepare-*"))
