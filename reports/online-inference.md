# Online inference

Модель `tm-0e83fa449b55`, режим dry-run: payload строится, но никуда не отправляется. Каждое окно получает запись со статусом и причиной.

## `online-ready.json`

| Статус → исход | Окон |
| --- | --- |
| predicted → not_publishable | 1 |
| predicted → would_publish | 1 |

| Символ | Статус | Причина | turn_event | Исход | Сигнал |
| --- | --- | --- | --- | --- | --- |
| SYM001 | predicted | — | TURN_DOWN | would_publish | SELL |
| SYM002 | predicted | — | NO_TURN | not_publishable | — |

## `online-warmup.json`

| Статус → исход | Окон |
| --- | --- |
| not_ready → not_publishable | 4 |

| Символ | Статус | Причина | turn_event | Исход | Сигнал |
| --- | --- | --- | --- | --- | --- |
| SYM003 | not_ready | not_enough_points | — | not_publishable | — |
| SYM004 | not_ready | stale_data | — | not_publishable | — |
| SYM005 | not_ready | missing_joined_rows | — | not_publishable | — |
| SYM006 | not_ready | warmup | — | not_publishable | — |

## `online-invalid.json`

| Статус → исход | Окон |
| --- | --- |
| invalid_input → not_publishable | 7 |

| Символ | Статус | Причина | turn_event | Исход | Сигнал |
| --- | --- | --- | --- | --- | --- |
| BAD001 | invalid_input | features.10.3: Input should be a valid number; features.10: Tuple should have... | — | not_publishable | — |
| BAD002 | invalid_input | features: Tuple should have at least 64 items after validation, not 63 | — | not_publishable | — |
| BAD003 | invalid_input | features.0: Tuple should have at most 6 items after validation, not 7; featur... | — | not_publishable | — |
| BAD004 | invalid_input | features.63.0: Input should be a valid number; features.63: Tuple should have... | — | not_publishable | — |
| BAD005 | invalid_input | item: Input tag 'BROKEN' found using 'state' does not match any of the expect... | — | not_publishable | — |
| BAD006 | invalid_input | signalBarrier: Extra inputs are not permitted | — | not_publishable | — |
| BAD007 | invalid_input | item: Value error, READY window must have 64 points | — | not_publishable | — |

## Правила

- `WARMUP`, `not_enough_points`, `stale_data`, `missing_joined_rows` — прогноз не формируется, сигнал не публикуется.
- Нечисловые значения, форма не 64×6, лишние поля, неизвестный state, неположительные цены — `invalid_input`, без публикации.
- Ошибка модели — `model_error` для каждого окна, остальные окна не страдают.
- Разрывы внутри READY-окна не интерполируются: их число — признак `missing_bars`.
- `NO_TURN` никогда не даёт BUY/SELL; одно окно не публикуется дважды.
