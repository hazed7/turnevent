from collections import Counter
from collections.abc import Mapping

from adapter.publisher import Decision
from reporting.markdown import document, section, table

_REASON_WIDTH = 80


def render_online_inference(runs: Mapping[str, tuple[Decision, ...]], model_version: str) -> str:
    return document(
        "Online inference",
        f"Модель `{model_version}`, режим dry-run: payload строится, но никуда не "
        "отправляется. Каждое окно получает запись со статусом и причиной.",
        *(_fixture(name, decisions) for name, decisions in runs.items()),
        section(
            "Правила",
            "- `WARMUP`, `not_enough_points`, `stale_data`, `missing_joined_rows` — "
            "прогноз не формируется, сигнал не публикуется.\n"
            "- Нечисловые значения, форма не 64×6, лишние поля, неизвестный state, "
            "неположительные цены — `invalid_input`, без публикации.\n"
            "- Ошибка модели — `model_error` для каждого окна, остальные окна не страдают.\n"
            "- Разрывы внутри READY-окна не интерполируются: их число — признак "
            "`missing_bars`.\n"
            "- `NO_TURN` никогда не даёт BUY/SELL; одно окно не публикуется дважды.",
        ),
    )


def _fixture(name: str, decisions: tuple[Decision, ...]) -> str:
    counts = Counter(f"{d.status} → {d.outcome}" for d in decisions)
    return section(
        f"`{name}`",
        table(["Статус → исход", "Окон"], sorted(counts.items())),
        table(
            ["Символ", "Статус", "Причина", "turn_event", "Исход", "Сигнал"],
            [
                [
                    d.symbol,
                    d.status,
                    _short(d.reason),
                    d.predicted_turn_event,
                    d.outcome,
                    d.payload.signal if d.payload else None,
                ]
                for d in decisions
            ],
        ),
    )


def _short(reason: str | None) -> str | None:
    if reason is None:
        return None
    first = reason.splitlines()[0].replace("|", "/")
    return first if len(first) <= _REASON_WIDTH else first[: _REASON_WIDTH - 3] + "..."
