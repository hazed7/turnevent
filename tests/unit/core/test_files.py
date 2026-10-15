import json
import stat
from pathlib import Path

import pytest

from core.files import write_json, write_text_atomic


def test_creates_parents_and_writes_pretty_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    write_json(target, {"b": 1, "a": [1, 2]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": [1, 2], "b": 1}


def test_written_files_are_world_readable(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    write_text_atomic(target, "x")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_overwrite_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    write_text_atomic(target, "v1")
    write_text_atomic(target, "v2")
    assert target.read_text(encoding="utf-8") == "v2"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_write_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.md"
    write_text_atomic(target, "original")

    def fail_replace(*_: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(target, "new")
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]
