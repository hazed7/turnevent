import io
import urllib.error
import urllib.request
from pathlib import Path
from typing import Self

import pytest

from core.errors import SourceError
from inference.source import RetryPolicy, fetch_envelope, read_fixture

URL = "http://ds-api.local/windows"


class _Body(io.BytesIO):
    def __enter__(self) -> Self:
        return self


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "error", {}, None)  # type: ignore[arg-type]


class _Server:
    def __init__(self, *outcomes: bytes | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[tuple[str, float]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, url: urllib.request.Request, /, *, timeout: float) -> _Body:
        self.requests.append((url.full_url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Body(outcome)


def test_transient_failures_are_retried_with_backoff() -> None:
    server = _Server(_http_error(503), urllib.error.URLError("reset"), b'{"ok": 1}')
    slept: list[float] = []
    body = fetch_envelope(URL, RetryPolicy(), opener=server, sleep=slept.append)
    assert body == b'{"ok": 1}'
    assert slept == [0.5, 1.0]
    assert server.requests == [(URL, 10.0)] * 3


@pytest.mark.parametrize("code", [400, 401, 404, 422])
def test_client_errors_are_not_retried(code: int) -> None:
    server = _Server(_http_error(code))
    with pytest.raises(SourceError, match=str(code)) as caught:
        fetch_envelope(URL, RetryPolicy(), opener=server, sleep=lambda _: None)
    assert server.calls == 1
    assert not caught.value.retryable


def test_rate_limiting_is_retried() -> None:
    server = _Server(_http_error(429), b"{}")
    assert fetch_envelope(URL, RetryPolicy(), opener=server, sleep=lambda _: None) == b"{}"


def test_giving_up_after_the_last_attempt() -> None:
    server = _Server(*[TimeoutError("slow")] * 3)
    slept: list[float] = []
    with pytest.raises(SourceError, match="unreachable") as caught:
        fetch_envelope(URL, RetryPolicy(attempts=3), opener=server, sleep=slept.append)
    assert server.calls == 3
    assert slept == [0.5, 1.0]
    assert caught.value.retryable


def test_backoff_is_capped() -> None:
    assert RetryPolicy(attempts=6, initial_delay_s=1.0, max_delay_s=4.0).delays() == [
        1.0,
        2.0,
        4.0,
        4.0,
        4.0,
    ]


def test_fixture_is_read_as_bytes(tmp_path: Path) -> None:
    path = tmp_path / "online.json"
    path.write_bytes(b"{}")
    assert read_fixture(path) == b"{}"
    with pytest.raises(SourceError, match="cannot read fixture"):
        read_fixture(tmp_path / "absent.json")
