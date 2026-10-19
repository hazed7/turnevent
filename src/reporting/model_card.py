from artifact.manifest import ModelManifest
from reporting.markdown import document, percent, section, table


def render_model_card(manifest: ModelManifest) -> str:
    metrics = manifest.metrics
    lag = manifest.prediction_lag_bars
    return document(
        f"Model card: {manifest.model_version}",
        section(
            "Назначение",
            "Классификация локального разворота тренда на минутных окнах MarketScreener DS "
            "API: `TURN_UP` (1), `NO_TURN` (0), `TURN_DOWN` (-1), плюс совместимый "
            "`turn_state` для текущего bundle. Исследовательский прототип интеграции; не "
            "торговая система и не инвестиционная рекомендация.",
        ),
        section(
            "Вход и выход",
            f"Вход — окно {manifest.window_length} × 6 (`rd_value, open, high, low, close, "
            f"volume`), таймфрейм {manifest.timeframe}, строки от старой к новой. Выход — "
            "классы и вероятности для `turn_event` и `turn_state`.\n\n"
            f"Сигнал на `windowEndTimestamp` означает разворот, подтверждённый на баре "
            f"`windowEndTimestamp − {lag} мин` (задержка детекции {lag} бара).",
        ),
        section(
            "Модель",
            "LogisticRegression со StandardScaler, `class_weight=balanced`, изотоническая "
            "калибровка вероятностей по времени. Порог TURN-классов "
            f"{manifest.event_turn_threshold} выбран на validation. Признаки "
            f"`{manifest.feature_schema_version}`: {len(manifest.feature_names)} шт., "
            f"нормализация — {manifest.normalization_spec}.",
        ),
        section(
            "Данные",
            f"Датасет `{manifest.dataset_version}` (sha256 `{manifest.dataset_sha256}`). "
            f"Разметка `{manifest.label_schema_version}`, генератор "
            f"`{manifest.label_generator_version}`. `signal_barrier` не используется нигде.",
        ),
        section(
            "Качество на holdout",
            table(
                ["", ""],
                [
                    ["Сэмплов", metrics.test_samples],
                    ["macro F1", metrics.macro_f1],
                    ["balanced accuracy", metrics.balanced_accuracy],
                    ["false-turn rate", percent(metrics.false_turn_rate)],
                    ["найдено разворотов", percent(metrics.detection_rate)],
                    ["медианная задержка, бары", metrics.median_delay_bars],
                    ["majority baseline, macro F1", metrics.majority_macro_f1],
                    ["rd_value rule baseline, macro F1", metrics.rd_value_rule_macro_f1],
                    ["turn_state, macro F1", metrics.state_macro_f1],
                    [
                        "log loss turn_event / turn_state",
                        f"{metrics.event_log_loss:.4f} / {metrics.state_log_loss:.4f}",
                    ],
                ],
            ),
        ),
        section(
            "Ограничения",
            "- Обучено на синтетике до получения approved-датасета: цифры проверяют пайплайн.\n"
            f"- Сигнал запаздывает на {lag} бара по построению.\n"
            "- Окна с разрывами не интерполируются: число пропущенных баров — признак.\n"
            "- Финальную публикацию решает decision layer MarketScreener.",
        ),
        section(
            "Окружение",
            table(
                ["Пакет", "Версия"],
                [["python", manifest.runtime.python], *manifest.runtime.packages.items()],
            ),
        ),
    )
