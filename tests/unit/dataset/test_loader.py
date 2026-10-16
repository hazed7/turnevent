from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from contracts.market import MODEL_INPUT_COLUMNS, Column
from core.errors import DataContractError
from core.serialization import sha256_file
from dataset.loader import SourceFormat, load_historical
from dataset.synthetic import SyntheticConfig, generate_synthetic

type WriteCsv = Callable[..., Path]

ROW = "60000,A,0.5,10,11,9,10.5,100,1"


def test_csv_loads_model_inputs_as_text_and_skips_signal_barrier(write_csv: WriteCsv) -> None:
    raw = load_historical(write_csv(ROW, " 120000 , A ,,10,11,9,10.5,100,-1"))
    assert tuple(raw.frame.columns) == MODEL_INPUT_COLUMNS
    assert raw.frame.loc[1, Column.TIMESTAMP] == "120000"
    assert raw.frame.loc[1, Column.SYMBOL] == "A"
    assert pd.isna(raw.frame.loc[1, Column.RD_VALUE])
    assert raw.source.signal_barrier_present
    assert raw.source.ignored_columns == (Column.SIGNAL_BARRIER,)


def test_source_is_fingerprinted_from_the_bytes_it_parsed(write_csv: WriteCsv) -> None:
    path = write_csv(ROW)
    source = load_historical(path).source
    assert source.sha256 == sha256_file(path)
    assert source.model_dump()["dataset_version"] == f"ds-{source.sha256[:12]}"
    assert source.source_format is SourceFormat.CSV


def test_utf8_bom_is_tolerated(write_csv: WriteCsv) -> None:
    path = write_csv(ROW)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_historical(path).source.header[0] == Column.TIMESTAMP


def test_parquet_records_arrow_types(tmp_path: Path) -> None:
    path = tmp_path / "history.parquet"
    config = SyntheticConfig(bars_per_symbol=64, n_symbols=1, plant_defects=False)
    generate_synthetic(config).to_parquet(path)
    raw = load_historical(path)
    assert raw.source.column_types[Column.TIMESTAMP] == "int64"
    assert raw.source.column_types[Column.CLOSE] == "double"
    assert tuple(raw.frame.columns) == MODEL_INPUT_COLUMNS


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("timestamp,symbol,rd_value,open,high,low,close,volume,open", "duplicate columns"),
        ("timestamp,symbol,rd_value,open,high,low,close", "missing required columns"),
        ("", "missing required columns"),
    ],
)
def test_bad_headers_are_rejected(write_csv: WriteCsv, header: str, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        load_historical(write_csv(header=header))


def test_non_utf8_csv_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    path.write_bytes(b"timestamp,symbol\n\xff\xfe\n")
    with pytest.raises(DataContractError, match="malformed CSV"):
        load_historical(path)


@pytest.mark.parametrize("ragged", [ROW + ",extra", ROW.rsplit(",", 1)[0]])
def test_ragged_rows_are_rejected(write_csv: WriteCsv, ragged: str) -> None:
    with pytest.raises(DataContractError, match="line 3 does not match the header"):
        load_historical(write_csv(ROW, ragged))


def test_corrupt_parquet_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "history.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(DataContractError, match="malformed parquet"):
        load_historical(path)


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="unsupported dataset format"):
        load_historical(tmp_path / "history.xlsx")


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="cannot read dataset"):
        load_historical(tmp_path / "absent.csv")
