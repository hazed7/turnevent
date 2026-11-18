from collections.abc import Mapping

from reporting.markdown import document, moment, percent, section, table
from training.metrics import ClassificationMetrics, EventMetrics, ProbabilityMetrics
from training.models import Algorithm
from training.pipeline import EvaluationReport, Predictor
from training.split import SplitManifest

ALGORITHM_NAMES = {
    Algorithm.LOGISTIC_REGRESSION: "LogisticRegression",
    Algorithm.HIST_GRADIENT_BOOSTING: "HistGradientBoosting",
}

_PREDICTOR_NAMES = {
    Predictor.MAJORITY: "majority (no-skill)",
    Predictor.RD_VALUE_RULE: "rule: rd_value extremum",
    Predictor.MODEL: "модель",
}


def render_evaluation(
    model_version: str,
    report: EvaluationReport,
    split: SplitManifest,
    by_lag: Mapping[int, EventMetrics],
) -> str:
    model = report.test_event[Predictor.MODEL]
    return document(
        "Evaluation",
        f"Модель `{model_version}`. Метка бара t сопоставляется с признаками бара "
        f"t + {report.prediction_lag_bars}: модель сигналит о развороте, подтверждённом "
        f"{report.prediction_lag_bars} бара назад. Порог для TURN-классов "
        f"{report.selected_turn_threshold} выбран на validation; test использован один раз.",
        section(
            "Разбиение",
            "Части идут последовательно по времени, без перемешивания. Сэмпл попадает в "
            "часть, только если все бары, от которых зависят его признаки и метка, лежат "
            "внутри её периода (purge); каждая следующая часть начинается через embargo "
            "после границы.",
            table(
                ["Часть", "Начало", "Конец", "Сэмплов", "Отсечено purge/embargo"],
                [
                    [p.part, moment(p.start_ms), moment(p.end_ms), p.samples, p.dropped_at_edges]
                    for p in split.periods
                ],
            ),
            f"Embargo: {split.embargo_ms // 60_000} баров.",
        ),
        section(
            "Holdout: модель против baseline-ов",
            table(
                [
                    "Предиктор",
                    "macro F1",
                    "balanced acc",
                    "false-turn rate",
                    "найдено разворотов",
                    "медианная задержка, бары",
                ],
                [
                    [
                        _PREDICTOR_NAMES[name],
                        m.macro_f1,
                        m.balanced_accuracy,
                        percent(m.false_turn_rate),
                        percent(m.detection.detection_rate),
                        m.detection.median_delay_bars,
                    ]
                    for name, m in report.test_event.items()
                ],
            ),
            "false-turn rate — доля ошибочных среди предсказанных разворотов; задержка "
            "считается от бара разворота до первого сигнала того же направления.",
            _rd_rule(report),
        ),
        section("Выбор алгоритма", *_algorithms(report)),
        section("Модель turn_event по классам", _per_class(model)),
        section("Confusion matrix turn_event (строки — истина)", _confusion(model)),
        section(
            "По временным сегментам",
            table(
                ["Начало", "Конец", "Сэмплов", "macro F1"],
                [
                    [moment(s.start_ms), moment(s.end_ms), s.samples, s.macro_f1]
                    for s in model.segments
                ],
            ),
        ),
        section(
            "Модель turn_state",
            _per_class(report.test_state),
            _confusion(report.test_state),
        ),
        section(
            "Калибровка вероятностей",
            _probabilities("turn_event", report.test_event_probabilities),
            _probabilities("turn_state", report.test_state_probabilities),
        ),
        section(
            "Момент сигнала",
            table(
                ["Лаг, бары", "macro F1", "balanced acc", "false-turn rate", "найдено"],
                [
                    [
                        lag,
                        m.macro_f1,
                        m.balanced_accuracy,
                        percent(m.false_turn_rate),
                        percent(m.detection.detection_rate),
                    ]
                    for lag, m in sorted(by_lag.items())
                ],
            ),
            "Лаг 0 — предсказание на самом баре разворота, когда подтверждающих баров ещё "
            "нет; лаг 3 — сигнал о подтверждённом развороте. Выбор лага — продуктовое "
            "решение, а не подгонка под метрику.",
        ),
        section(
            "Ограничения",
            f"- Классов TURN_* около {percent(model.true_share['TURN_UP'])} каждого; macro F1 "
            "сильно зависит от них.\n"
            "- Разметка и признаки зафиксированы до обучения; test не участвовал ни в одном "
            "выборе.",
        ),
    )


def _per_class(metrics: ClassificationMetrics) -> str:
    return table(
        ["Класс", "precision", "recall", "F1", "support"],
        [[name, m.precision, m.recall, m.f1, m.support] for name, m in metrics.per_class.items()],
    )


def _confusion(metrics: ClassificationMetrics) -> str:
    names = list(metrics.per_class)
    return table(
        ["истина \\ прогноз", *names],
        [[names[i], *row] for i, row in enumerate(metrics.confusion_matrix)],
    )


def _probabilities(title: str, metrics: ProbabilityMetrics) -> str:
    return (
        table(
            [title, "средняя вероятность", "реальная доля"],
            [
                [name, metrics.mean_probability[name], metrics.true_share[name]]
                for name in metrics.mean_probability
            ],
        )
        + f"\n\nlog loss {metrics.log_loss:.4f}, Brier {metrics.brier_score:.4f}."
    )


def _rd_rule(report: EvaluationReport) -> str:
    rule = report.selected_rd_rule
    low, high = ("TURN_UP", "TURN_DOWN") if rule.direction == 1 else ("TURN_DOWN", "TURN_UP")
    return (
        f"Правило rd_value: значение {rule.reference_lag} бар(а) назад отклонилось от среднего "
        f"за 20 баров не меньше чем на {rule.min_zscore} σ, а с тех пор rd_value вернулось "
        f"назад не меньше чем на {rule.min_rebound} σ. Минимум rd_value → {low}, "
        f"максимум → {high}. Все четыре параметра выбраны по macro F1 на validation."
    )


def _algorithms(report: EvaluationReport) -> tuple[str, ...]:
    selected = ALGORITHM_NAMES[report.selected_algorithm]
    intro = (
        "Основной алгоритм — LogisticRegression; альтернатива — градиентный бустинг "
        "деревьев (HistGradientBoosting). Причина эксперимента: метка — пороговое правило "
        "по ценам (экстремум окна и откат относительно ATR), а линейная модель не выражает "
        "такие условия «и/если». Оба алгоритма обучены на train с одинаковыми признаками, "
        "весами классов и калибровкой; порог TURN-классов подобран на validation для каждого. "
        f"В bundle попадает алгоритм с лучшим macro F1 на validation: **{selected}**. "
        "Test показан для справки и в выборе не участвовал."
    )
    rows = [
        [
            ALGORITHM_NAMES[algorithm],
            report.validation_by_algorithm[algorithm].macro_f1,
            test.macro_f1,
            test.balanced_accuracy,
            percent(test.false_turn_rate),
            percent(test.detection.detection_rate),
            "да" if algorithm is report.selected_algorithm else "",
        ]
        for algorithm, test in report.test_by_algorithm.items()
    ]
    return (
        intro,
        table(
            [
                "Алгоритм",
                "validation macro F1",
                "test macro F1",
                "test balanced acc",
                "test false-turn rate",
                "test найдено",
                "выбран",
            ],
            rows,
        ),
    )
