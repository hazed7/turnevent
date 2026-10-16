import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def load_fixture() -> Callable[[str], Any]:
    def load(name: str) -> Any:  # noqa: ANN401
        return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))

    return load
