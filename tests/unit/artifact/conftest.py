from pathlib import Path

import pytest


@pytest.fixture
def bundle_dir(trained_bundle_dir: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "bundle"
    copy.mkdir()
    for file in trained_bundle_dir.iterdir():
        (copy / file.name).write_bytes(file.read_bytes())
    return copy
