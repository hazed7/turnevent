from collections.abc import Mapping

from reporting.markdown import document, moment, percent, section, table
from training.metrics import ClassificationMetrics, EventMetrics, ProbabilityMetrics
from training.pipeline import EvaluationReport, Predictor
from training.split import SplitManifest

_PREDICTOR_NAMES = {
    Predictor.MAJORITY: "majority (no-skill)",
    Predictor.RD_VALUE_RULE: "rule: rd_value cross",
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
        ),
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
            "- Метрики посчитаны на синтетическом датасете и проверяют пайплайн, а не "
            "качество на рынке.\n"
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
