В папку `dataset` надо закинуть train.csv и test.csv с Яндекс.Диска.

Доступный контекст по кейсу для принятия решений:
* `docs/CASE.md` - полноценное полное описание кейса 
* `docs/DATASET.md` - описание датасета от заказчика
* `docs/PREVIEW.md` - предварительное описание кейса на сайте
* `docs/REMARKS.md` - вопросы из общего чата, адресованные заказчику
* `docs/THOUGHTS.md` - человеческие размышления по внутренним планы по разработке
* `docs/GEO-HOLDOUT.md` - отчёт-исследование: прогноз загрузки нововведённой линии
* `docs/QA session/*.md` - расшифровка и саммари по Q&A сессии с заказчиком

Отдельные результаты раздельного прототипирования:
* `weather/` - данные по прогнозу погоды в Москве на 2025 год
* `traffic/` - данные по загруженности дорог в Москве на 2025 год
* `factors/` - согласование данных по погоде и загрузке с трамвайными маршрутами
* `geo/` - гео-признаки маршрутов из OSM и холдаут по маршрутам: можно ли прогнозировать линию, которой не было в обучении (отчёт - `docs/GEO-HOLDOUT.md`)
* `input/calendar/` - производственный календарь на 2025 год
* `service/` - Go-сервис прогноза: CatBoost и CNN, условия, выгрузка (см. `service/README.md`)
* `dashboard/` - дашборд диспетчера на React: карта, графики, условия, сравнение моделей (см. `dashboard/README.md`)

Результаты тестирования базовых моделей:
* `catboost_baseline.ipynb` - гарантированно дешёвое воспроизводимое и интерпретируемое решение кейса на CatBoost с 0.871 WAPE-score
* `harmonic_baseline.ipynb` - гармоническая модель с WAPE-score 0.867, при 50/50 бленде с катбустом даёт 0.874.
* `mlp_baseline.ipynb` - простенькая MLP нейросеть

## Запуск стенда

Стенд - один контейнер: Go-сервис отдаёт и API, и собранный дашборд. Нужен только Docker
с Compose; бандлы моделей лежат в репозитории, датасет для запуска не нужен.

```bash
docker compose up -d --build                          # http://127.0.0.1:8090
STAND_AUTH=jury:secret docker compose up -d --build   # то же под basic auth
docker compose logs server                            # обе строки сверки с эталоном
curl http://127.0.0.1:8090/healthz
```

Образ собирается в три этапа (`Dockerfile`): pnpm и Node собирают `dashboard/` в статику,
`golang:1.24-bookworm` скачивает `libcatboostmodel` и собирает сервер, `debian:bookworm-slim`
получает сервер, библиотеку, бандлы, календари, геометрию и статику. Node в итоговом образе
нет. При старте сервер сверяет обе модели с эталоном и не запускается, если сверка не прошла.

Без Docker, для разработки. Бэкенд (Go 1.24), из `service/`:

```bash
bash scripts/fetch_libs.sh
go run ./cmd/server                                   # API на http://localhost:8080
```

Фронтенд с горячей перезагрузкой (Node 24, pnpm 11), из `dashboard/`; Vite проксирует
`/api`, `/geo` и `/healthz` на `localhost:8080`:

```bash
pnpm install
pnpm dev                                              # http://localhost:5173
```

Собранный дашборд без контейнера: `pnpm build` в `dashboard/`, затем из `service/`
`go run ./cmd/server -static ../dashboard/dist` - интерфейс на http://localhost:8080.
