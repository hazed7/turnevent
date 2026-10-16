from collections.abc import Callable
from pathlib import Path

import pytest

from contracts.market import HISTORICAL_COLUMNS


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[..., Path]:
    def write(*rows: str, header: str = ",".join(HISTORICAL_COLUMNS)) -> Path:
        path = tmp_path / "history.csv"
        path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return path

    return write
