from pathlib import Path

import pytest

from core.config import load_yaml_config
from core.errors import ConfigError
from core.frozen import FrozenModel


class Labels(FrozenModel):
    left: int
    right: int


class SampleConfig(FrozenModel):
    labels: Labels
    symbols: tuple[str, ...] = ()
    seed: int = 42


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_config_and_coerces_yaml_lists(tmp_path: Path) -> None:
    path = _write(tmp_path, "labels: {left: 3, right: 3}\nsymbols: [A, B]\n")
    assert load_yaml_config(path, SampleConfig) == SampleConfig(
        labels=Labels(left=3, right=3), symbols=("A", "B")
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must be a mapping, got NoneType"),
        ("labels: [unclosed", "malformed YAML"),
        ("- 1\n- 2\n", "must be a mapping, got list"),
        ("labels: {left: three, right: 3}\n", "invalid config"),
        ("labels: {left: 3, right: 3}\nextra: 1\n", "invalid config"),
    ],
)
def test_bad_configs_raise_config_error(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_yaml_config(_write(tmp_path, text), SampleConfig)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_yaml_config(tmp_path / "absent.yaml", SampleConfig)
