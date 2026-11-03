import json
import os
import stat
from pathlib import Path

import pytest

from core.files import publish_exclusive, write_json, write_text_atomic


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


def test_publish_exclusive_links_every_pair(tmp_path: Path) -> None:
    first, second = tmp_path / "a.tmp", tmp_path / "b.tmp"
    first.write_text("a")
    second.write_text("b")
    publish_exclusive((first, tmp_path / "a"), (second, tmp_path / "b"))
    assert (tmp_path / "a").read_text() == "a"
    assert (tmp_path / "b").read_text() == "b"


def test_publish_exclusive_never_replaces_existing_target(tmp_path: Path) -> None:
    source, target = tmp_path / "new.tmp", tmp_path / "out"
    source.write_text("new")
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        publish_exclusive((source, target))
    assert target.read_text() == "keep"


@pytest.mark.parametrize("error", [FileExistsError, KeyboardInterrupt])
def test_publish_exclusive_rolls_back_on_any_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    first, second = tmp_path / "a.tmp", tmp_path / "b.tmp"
    first.write_text("a")
    second.write_text("b")
    link = os.link

    def failing_link(source: Path, target: Path) -> None:
        if target == tmp_path / "b":
            raise error
        link(source, target)

    monkeypatch.setattr("core.files.os.link", failing_link)
    with pytest.raises(error):
        publish_exclusive((first, tmp_path / "a"), (second, tmp_path / "b"))
    assert not (tmp_path / "a").exists()


def test_publish_exclusive_rollback_spares_replaced_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "a.tmp", tmp_path / "b.tmp"
    first.write_text("a")
    second.write_text("b")
    link = os.link

    def racing_link(source: Path, target: Path) -> None:
        if target == tmp_path / "b":
            (tmp_path / "a").unlink()
            (tmp_path / "a").write_text("other writer")
            raise FileExistsError
        link(source, target)

    monkeypatch.setattr("core.files.os.link", racing_link)
    with pytest.raises(FileExistsError):
        publish_exclusive((first, tmp_path / "a"), (second, tmp_path / "b"))
    assert (tmp_path / "a").read_text() == "other writer"
