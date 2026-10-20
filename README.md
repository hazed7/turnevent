Детектор локальных разворотов тренда на минутных свечах. Три класса:
`TURN_UP`, `NO_TURN`, `TURN_DOWN`. Внутри разметка исторических данных, обучение и оценка
модели, сборка bundle, инференс по окнам DS API и выдача сигналов в их формате

## Запуск

Понадобятся `uv` и `just`

```bash
just install    # зависимости и pre-commit
just pipeline   # синтетика -> аудит -> обучение -> прогон
just check      # линтеры, mypy, тесты
```

После `just pipeline` в `reports/` лежат отчёты, в `manifests/` манифесты, а модель
сохраняется в `artifacts/<model_version>/`. Версия модели считается из содержимого, так
что на тех же данных и конфиге получается тот же bundle.

## Команды

Отдельные шаги запускаются через `just cli <команда>`, параметры берутся из
`configs/pipeline.yaml`.

- `synthetic` генерирует синтетический датасет
- `audit` пишет `reports/data-audit.md`, манифест датасета и определение разметки
- `train` обучает модель, сохраняет бандл и пишет `evaluation.md` и `model-card.md`
- `infer --fixture fixtures/online-ready.json` прогоняет один конверт из файла
- `infer --url <endpoint> --mode shadow` берёт конверт с API и пишет сигналы в shadow
- `smoke` прогоняет все фикстуры и пишет `reports/online-inference.md`
- `all` делает всё по очереди

Результаты `infer` складываются в `runs/`: в `telemetry.jsonl` попадает каждое окно, в
`outbox.jsonl` отправленные сигналы, а `ledger.jsonl` следит, чтобы одно окно не ушло
дважды.

## Данные

Путь к датасету задаётся в `paths.dataset`. Подходит CSV или parquet с колонками
`timestamp, symbol, rd_value, open, high, low, close, volume, signal_barrier`

## Структура

```
src/
  core/        общие утилиты: хэши, запись файлов, конфиги
  contracts/   форматы данных: колонки, классы, конверт API, payload
  dataset/     загрузка, чистка, аудит, синтетика
  labels/      разметка разворотов
  features/    признаки, одни и те же для обучения и онлайна
  artifact/    классификатор и бандл модели
  training/    сплит, бейзлайны, модель, метрики, экспорт
  inference/   обработка окон API
  adapter/     сигналы BUY/SELL, дедупликация, shadow
  reporting/   отчёты
  cli/         команды
```

## Пара важных моментов

Модель сигналит о развороте, который уже подтвердился, то есть с задержкой в 3 бара:
сигнал на `windowEndTimestamp` относится к бару `windowEndTimestamp − 3 мин`. Задержка
настраивается в `training.prediction_lag_bars`.
В `evaluation.md` есть сравнение с задержкой.

Train, validation и test идут по времени, между ними стоят purge и embargo,
порог подбирается на валидации, а на test модель смотрит один раз.

Ошибки, неготовые окна и `NO_TURN` никогда не превращаются в BUY/SELL
