from contracts.market import WINDOW_LENGTH, Column
from dataset.audit import DataAudit
from labels.definition import LabelSummary
from labels.generator import LabelConfig
from labels.sensitivity import SensitivityRow
from reporting.markdown import document, moment, percent, section, table


def render_data_audit(
    audit: DataAudit,
    labels: LabelSummary,
    label_config: LabelConfig,
    sensitivity: tuple[SensitivityRow, ...],
) -> str:
    source = audit.source
    return document(
        "Data audit",
        table(
            ["", ""],
            [
                ["Файл", source.file_name],
                ["sha256", f"`{source.sha256}`"],
                ["dataset_version", source.dataset_version],
                ["Формат", source.source_format],
                ["Период", f"{moment(audit.period_start_ms)} — {moment(audit.period_end_ms)}"],
                ["Строк (сырые / чистые)", f"{audit.rows_raw} / {audit.rows_clean}"],
                ["Символов", len(audit.symbols)],
            ],
        ),
        section(
            "Колонки источника",
            table(
                ["Колонка", "Тип в источнике", "Загружена"],
                [
                    [name, kind, "да" if name in audit.loaded_columns else "нет"]
                    for name, kind in source.column_types.items()
                ],
            ),
        ),
        section(
            "signal_barrier",
            f"В источнике: **{'есть' if source.signal_barrier_present else 'нет'}**. "
            f"Загружены только колонки {', '.join(f'`{c}`' for c in audit.loaded_columns)}: "
            f"`{Column.SIGNAL_BARRIER}` не читается в память, поэтому не может попасть в "
            "признаки, разметку, веса или фильтры. Это проверяет property-тест: любые "
            "значения `signal_barrier` дают побайтово одинаковый чистый датасет.",
        ),
        section(
            "Чистка",
            f"Политика `{audit.cleaning.policy_version}`. Каждая строка отброшена по первой "
            "нарушенной причине.",
            table(
                ["Причина", "Строк"],
                [[reason, count] for reason, count in audit.cleaning.dropped.items()],
            ),
            table(
                ["Колонка", "Пропуски", "Нечисловые или бесконечные"],
                [
                    [column, missing, audit.cleaning.invalid_values[column]]
                    for column, missing in audit.cleaning.missing_values.items()
                ],
            ),
            f"Немонотонных шагов времени в исходном порядке: "
            f"{audit.cleaning.non_monotonic_steps}. Меток вне минутной сетки (сохранены): "
            f"{audit.cleaning.off_grid_timestamps}.",
        ),
        section(
            "Периодичность и окна",
            table(
                ["", ""],
                [
                    ["Ожидаемый шаг, мс", audit.periodicity.expected_interval_ms],
                    ["Медианный шаг, мс", audit.periodicity.median_step_ms],
                    ["Доля регулярных шагов", percent(audit.periodicity.regular_step_share)],
                    ["Разрывов", audit.periodicity.gaps],
                    ["Пропущенных баров", audit.periodicity.missing_bars],
                    ["Нерегулярных шагов", audit.periodicity.irregular_steps],
                    ["Длина окна", WINDOW_LENGTH],
                    ["Доступных окон", audit.available_windows],
                ],
            ),
            table(
                ["Символ", "Строк", "Разрывов", "Пропущено баров", "Окон"],
                [[s.symbol, s.rows, s.gaps, s.missing_bars, s.windows] for s in audit.symbols],
            ),
        ),
        section(
            "Разметка",
            f"Генератор: окно {label_config.left_bars}+{label_config.right_bars}, разнос "
            f"{label_config.min_separation_bars} бара, порог "
            f"{label_config.threshold_coefficient} × ATR({label_config.scale_window_bars}).",
            table(["Статус", "Строк"], [[status, n] for status, n in labels.by_status.items()]),
            table(
                ["Класс turn_event", "Строк", "Доля"],
                [
                    [name, n, percent(n / labels.labelled if labels.labelled else None)]
                    for name, n in labels.turn_event.items()
                ],
            ),
            table(
                ["Класс turn_state", "Строк", "Доля"],
                [
                    [name, n, percent(n / labels.labelled if labels.labelled else None)]
                    for name, n in labels.turn_state.items()
                ],
            ),
        ),
        section(
            "Чувствительность разметки к параметрам",
            table(
                ["Параметр", "Значение", "TURN_UP", "TURN_DOWN", "NO_TURN", "Размечено"],
                [
                    [
                        row.parameter,
                        str(row.value),
                        row.summary.turn_event["TURN_UP"],
                        row.summary.turn_event["TURN_DOWN"],
                        row.summary.turn_event["NO_TURN"],
                        row.summary.labelled,
                    ]
                    for row in sensitivity
                ],
            ),
        ),
        section(
            "Окружение",
            table(
                ["Пакет", "Версия"],
                [["python", audit.runtime.python], *audit.runtime.packages.items()],
            ),
            f"`uv.lock` sha256: `{audit.runtime.lockfile_sha256}`",
        ),
    )
