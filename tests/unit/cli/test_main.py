from pathlib import Path

import pytest

from cli.__main__ import main


def test_commands_report_what_they_did(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_file), "all"]) == 0
    assert "pipeline finished: tm-" in capsys.readouterr().out

    fixture = Path("fixtures/online-ready.json")
    assert main(["--config", str(config_file), "infer", "--fixture", str(fixture)]) == 0
    assert "publish" in capsys.readouterr().out

    for command in ("synthetic", "audit", "train", "smoke"):
        assert main(["--config", str(config_file), command]) == 0
        assert capsys.readouterr().out.strip()


def test_errors_exit_with_a_message(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(config_file), "infer", "--fixture", "missing.json"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_missing_config_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "absent.yaml"), "audit"]) == 1
    assert "cannot read config" in capsys.readouterr().err
