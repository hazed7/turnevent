import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from pydantic import PositiveFloat, PositiveInt

from core.errors import SourceError
from core.frozen import FrozenModel


class RetryPolicy(FrozenModel):
    attempts: PositiveInt = 4
    initial_delay_s: PositiveFloat = 0.5
    max_delay_s: PositiveFloat = 8.0
    timeout_s: PositiveFloat = 10.0

    def delays(self) -> list[float]:
        return [
            min(self.initial_delay_s * 2**attempt, self.max_delay_s)
            for attempt in range(self.attempts - 1)
        ]


class Response(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> object: ...


class Opener(Protocol):
    def __call__(self, url: urllib.request.Request, /, *, timeout: float) -> Response: ...


def read_fixture(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceError(f"cannot read fixture {path}: {exc}", retryable=False) from exc


def fetch_envelope(
    url: str,
    policy: RetryPolicy,
    *,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
    delays = iter(policy.delays())
    while True:
        try:
            with opener(request, timeout=policy.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            error = SourceError(f"DS API answered {exc.code}", retryable=_retryable(exc.code))
        except (urllib.error.URLError, TimeoutError) as exc:
            error = SourceError(f"DS API unreachable: {exc}", retryable=True)
        delay = next(delays, None)
        if not error.retryable or delay is None:
            raise error
        sleep(delay)


def _retryable(status: int) -> bool:
    return status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.TOO_MANY_REQUESTS
