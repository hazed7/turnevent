from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f"""
paths:
  dataset: {tmp_path / "data" / "history.csv"}
  fixtures: {PROJECT_ROOT / "fixtures"}
  manifests: {tmp_path / "manifests"}
  reports: {tmp_path / "reports"}
  artifacts: {tmp_path / "artifacts"}
  runs: {tmp_path / "runs"}
synthetic:
  n_symbols: 2
  bars_per_symbol: 3000
""",
        encoding="utf-8",
    )
    return path
