from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

type Cell = str | int | float | None


def table(headers: Sequence[str], rows: Iterable[Sequence[Cell]]) -> str:
    lines = [_row(headers), _row(["---"] * len(headers))]
    lines += [_row([cell(value) for value in row]) for row in rows]
    return "\n".join(lines)


def cell(value: Cell) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def moment(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def document(title: str, *sections: str) -> str:
    return "\n\n".join([f"# {title}", *sections]) + "\n"


def section(title: str, *blocks: str) -> str:
    return "\n\n".join([f"## {title}", *blocks])


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"
