# Model card: tm-0e83fa449b55

## Назначение

Классификация локального разворота тренда на минутных окнах MarketScreener DS API: `TURN_UP` (1), `NO_TURN` (0), `TURN_DOWN` (-1), плюс совместимый `turn_state` для текущего bundle. Исследовательский прототип интеграции; не торговая система и не инвестиционная рекомендация.

## Вход и выход

Вход — окно 64 × 6 (`rd_value, open, high, low, close, volume`), таймфрейм 1m, строки от старой к новой. Выход — классы и вероятности для `turn_event` и `turn_state`.

Сигнал на `windowEndTimestamp` означает разворот, подтверждённый на баре `windowEndTimestamp − 3 мин` (задержка детекции 3 бара).

## Модель

LogisticRegression со StandardScaler, `class_weight=balanced`, изотоническая калибровка вероятностей по времени. Порог TURN-классов 0.36 выбран на validation. Признаки `1.1.0`: 35 шт., нормализация — stateless and per window, nothing is fitted on training data: prices enter as log ratios or fractions of the last close, volume as log1p ratios to the window mean, rd_value raw plus a z-score within the window.

## Данные

Датасет `ds-d255dd3aa3c5` (sha256 `d255dd3aa3c5d65ce12ccf1591feb6a69f20c0df02b328c4697f7cef3683a79b`). Разметка `1.0.0`, генератор `1.0.0`. `signal_barrier` не используется нигде.

## Качество на holdout

|  |  |
| --- | --- |
| Сэмплов | 11135 |
| macro F1 | 0.860 |
| balanced accuracy | 0.905 |
| false-turn rate | 26.5% |
| найдено разворотов | 86.4% |
| медианная задержка, бары | 3.000 |
| majority baseline, macro F1 | 0.326 |
| rd_value rule baseline, macro F1 | 0.354 |
| turn_state, macro F1 | 0.578 |
| log loss turn_event / turn_state | 0.0500 / 0.7900 |

## Ограничения

- Обучено на синтетике до получения approved-датасета: цифры проверяют пайплайн.
- Сигнал запаздывает на 3 бара по построению.
- Окна с разрывами не интерполируются: число пропущенных баров — признак.
- Финальную публикацию решает decision layer MarketScreener.

## Окружение

| Пакет | Версия |
| --- | --- |
| python | 3.13.13 |
| joblib | 1.6.0 |
| numpy | 2.5.4 |
| packaging | 26.3 |
| pandas | 3.0.6 |
| pyarrow | 26.0.0 |
| pydantic | 2.14.1 |
| pyyaml | 6.0.3 |
| scikit-learn | 1.9.1 |
