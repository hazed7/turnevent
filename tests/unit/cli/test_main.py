from pathlib import Path

import pytest

from cli.__main__ import main


def test_commands_report_what_they_did(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_file), "synthetic"]) == 0
    assert "dataset written" in capsys.readouterr().out
    assert main(["--config", str(config_file), "all"]) == 0
    assert "pipeline finished: tm-" in capsys.readouterr().out

    fixture = Path("fixtures/online-ready.json")
    assert main(["--config", str(config_file), "infer", "--fixture", str(fixture)]) == 0
    assert "publish" in capsys.readouterr().out

    for command in ("audit", "train", "smoke"):
        assert main(["--config", str(config_file), command]) == 0
        assert capsys.readouterr().out.strip()


def test_errors_exit_with_a_message(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(config_file), "infer", "--fixture", "missing.json"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_missing_config_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "absent.yaml"), "audit"]) == 1
    assert "cannot read config" in capsys.readouterr().err


def test_all_requires_prepared_dataset(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_file), "all"]) == 1
    assert "cannot read dataset" in capsys.readouterr().err
    assert not (config_file.parent / "data" / "history.csv").exists()


def test_demo_is_explicit_and_refuses_to_overwrite(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_file), "demo"]) == 0
    assert "demo finished: tm-" in capsys.readouterr().out
    assert main(["--config", str(config_file), "demo"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
