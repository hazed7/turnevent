import json
import os
from pathlib import Path

from core.errors import LedgerError

type WindowKey = tuple[str, int]


class PublicationLedger:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._published: set[WindowKey] = set(self._read(path)) if path else set()

    def __contains__(self, key: WindowKey) -> bool:
        return key in self._published

    def record(self, key: WindowKey) -> None:
        if self._path is not None:
            self._append(self._path, key)
        self._published.add(key)

    @staticmethod
    def _read(path: Path) -> list[WindowKey]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            return [(entry["symbol"], int(entry["timestamp"])) for entry in map(json.loads, lines)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LedgerError(f"unreadable publication ledger {path}: {exc}") from exc

    @staticmethod
    def _append(path: Path, key: WindowKey) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"symbol": key[0], "timestamp": key[1]}, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
