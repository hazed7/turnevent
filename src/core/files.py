import contextlib
import os
import tempfile
from pathlib import Path

from core.serialization import to_pretty_json

_FILE_MODE = 0o644


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.chmod(_FILE_MODE)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    _fsync_dir(path.parent)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json(path: Path, value: object) -> None:
    write_text_atomic(path, to_pretty_json(value))


def publish_exclusive(*pairs: tuple[Path, Path]) -> None:
    for source, _ in pairs:
        _fsync_file(source)
    published: list[tuple[Path, Path]] = []
    try:
        for source, target in pairs:
            os.link(source, target)
            published.append((source, target))
    except BaseException:
        for source, target in reversed(published):
            _unlink_if_same(target, source)
        raise
    for directory in {target.parent for _, target in pairs}:
        _fsync_dir(directory)


def _unlink_if_same(path: Path, original: Path) -> None:
    with contextlib.suppress(OSError):
        if path.samefile(original):
            path.unlink()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
