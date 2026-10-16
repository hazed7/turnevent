import csv
import io
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import computed_field

from contracts.market import MODEL_INPUT_COLUMNS, Column
from core.errors import DataContractError
from core.frozen import FrozenModel
from core.serialization import sha256_bytes


class SourceFormat(StrEnum):
    CSV = auto()
    PARQUET = auto()


class SourceInfo(FrozenModel):
    file_name: str
    sha256: str
    source_format: SourceFormat
    header: tuple[str, ...]
    column_types: Mapping[str, str]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_version(self) -> str:
        return f"ds-{self.sha256[:12]}"

    @property
    def signal_barrier_present(self) -> bool:
        return Column.SIGNAL_BARRIER in self.header

    @property
    def ignored_columns(self) -> tuple[str, ...]:
        return tuple(column for column in self.header if column not in MODEL_INPUT_COLUMNS)


@dataclass(frozen=True, slots=True)
class RawDataset:
    frame: pd.DataFrame
    source: SourceInfo


@dataclass(frozen=True, slots=True)
class _Table:
    header: tuple[str, ...]
    column_types: Mapping[str, str]
    frame: pd.DataFrame


def load_historical(path: Path) -> RawDataset:
    source_format = _detect_format(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DataContractError(f"cannot read dataset {path}: {exc}") from exc

    read = _read_csv if source_format is SourceFormat.CSV else _read_parquet
    table = read(data)
    source = SourceInfo(
        file_name=path.name,
        sha256=sha256_bytes(data),
        source_format=source_format,
        header=table.header,
        column_types=table.column_types,
    )
    return RawDataset(frame=_as_text(table.frame), source=source)


def _detect_format(path: Path) -> SourceFormat:
    try:
        return SourceFormat(path.suffix.lower().removeprefix("."))
    except ValueError:
        supported = ", ".join(f".{member}" for member in SourceFormat)
        raise DataContractError(
            f"unsupported dataset format {path.suffix!r}; expected one of: {supported}"
        ) from None


def _read_csv(data: bytes) -> _Table:
    try:
        text = data.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text))

        header = tuple(next(rows, ()))
        _check_header(header)

        ragged = next((rows.line_num for row in rows if len(row) != len(header)), None)
        if ragged is not None:
            raise DataContractError(f"malformed CSV: line {ragged} does not match the header")

        frame = pd.read_csv(
            io.StringIO(text),
            usecols=list(MODEL_INPUT_COLUMNS),
            dtype=str,
            keep_default_na=False,
        )
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError) as exc:
        raise DataContractError(f"malformed CSV: {exc}") from exc

    return _Table(header=header, column_types=dict.fromkeys(header, "text"), frame=frame)


def _read_parquet(data: bytes) -> _Table:
    try:
        parquet = pq.ParquetFile(pa.BufferReader(data))
        schema = parquet.schema_arrow

        header = tuple(schema.names)
        _check_header(header)

        table = parquet.read(columns=list(MODEL_INPUT_COLUMNS))
    except (pa.ArrowException, OSError) as exc:
        raise DataContractError(f"malformed parquet: {exc}") from exc

    return _Table(
        header=header,
        column_types={field.name: str(field.type) for field in schema},
        frame=table.to_pandas(),
    )


def _check_header(header: tuple[str, ...]) -> None:
    duplicated = sorted(column for column, count in Counter(header).items() if count > 1)
    if duplicated:
        raise DataContractError(f"duplicate columns in header: {duplicated}")

    missing = [column for column in MODEL_INPUT_COLUMNS if column not in header]
    if missing:
        raise DataContractError(f"missing required columns: {missing}")


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: frame[column].astype("str").str.strip().replace({"": pd.NA})
            for column in MODEL_INPUT_COLUMNS
        },
    )
