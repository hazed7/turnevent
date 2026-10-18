from pathlib import Path

import pytest

from adapter.ledger import PublicationLedger
from core.errors import LedgerError


def test_published_windows_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    PublicationLedger(path).record(("SYM001", 60_000))
    reopened = PublicationLedger(path)
    assert ("SYM001", 60_000) in reopened
    assert ("SYM001", 120_000) not in reopened


def test_ledger_without_a_file_lives_in_memory() -> None:
    ledger = PublicationLedger()
    ledger.record(("SYM001", 60_000))
    assert ("SYM001", 60_000) in ledger


def test_corrupt_ledger_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"symbol": "SYM001"}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="unreadable publication ledger"):
        PublicationLedger(path)
