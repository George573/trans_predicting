# Dry-plan: приведение репозитория к модульной структуре

Статус: план, ничего не выполнено. Зафиксировано 27.09.2026.

Цель - разложить проект по модулям в порядке, который требует кейс: приём и нормализация
данных -> признаки и геопривязка -> ML-прогноз и агрегация -> API -> frontend. Один
репозиторий, одна ветка, одна команда на каждый шаг.

## Текущее состояние

- Проект разнесён по трём веткам: `main` (ML, исследования, внешние данные), `feature/cnn`
  (основная модель и Go-сервис с CNN), `go-inference-stand` (Go-стенд на CatBoost).
- В корне `main` 12 Python-модулей и 4 ноутбука, модули импортируют друг друга напрямую
  (`from backtest import ...`). Нет пакета, нет `pyproject.toml` или `requirements.txt`.
- Пути к данным зашиты относительно корня (`"dataset/labels/..."`, `"input/calendar/2025.xml"`)
  примерно в 20 местах.
- Сборщики внешних данных (`weather/`, `traffic/`, `events/`, `factors/`, `geo/`) - отдельные
  папки со скриптами, данными и README вперемешку.
- Производственный календарь в трёх копиях: `input/calendar/`, `src/tram_forecast/calendars/`
  и `example_holidays/` в CNN-ветке.
- В git лежат бинарные артефакты на 32 МБ: 46 сабмитов, parquet-прогнозы, модели; `.git` весит
  82 МБ.
- В `docs/` смешаны рабочие заметки, спеки и отчёты.

## Целевая структура

```
README.md                    <- final/readme.md
docs/                        <- final/docs (+ img/, diagrams/)
  notes/                     <- рабочие материалы: CASE, DATASET, PREVIEW, REMARKS,
                                THOUGHTS, QA session, CONTEXT, MODEL, SPEC-*, RESEARCH-*,
                                GEO-HOLDOUT, этот файл
api/                         <- openapi.yaml + examples (контракт, без изменений)
pyproject.toml               <- один Python-проект, зависимости по группам
src/
  tram_data/                 1. приём и нормализация, внешние источники
    paths.py                    все пути к данным от корня репозитория, переопределение через env
    labels.py                   сырые валидации -> почасовая сетка
    calendar.py                 единый загрузчик производственного календаря для обеих моделей
    sources/weather.py, traffic.py, events.py, geo.py
    factors.py                  сопоставление факторов с маршрутами
    effects/                    замеры эффектов (бывшие check_*.py)
  tram_ml/                   2-3. признаки + классическая модель
    features.py, metrics.py, backtest.py
    models/catboost.py, harmonic.py, mlp.py
    ensemble.py, calibration.py, postprocess.py
    cli.py                      python -m tram_ml build-submission
  tram_forecast/             3. CNN из feature/cnn, без изменений внутри
    ...                         data, model, train, predict, runner, head_export, cli
research/                    исследования, не продакшн
  horizon/                      бывшие experiments/*.py
  geo/                          loro_experiment.py
  notebooks/                    analysis, *_baseline.ipynb, train.ipynb от CNN
  archive/                      отклонённые подходы
service/                     4. Go-модуль, одна точка сборки
  cmd/server, cmd/load, cmd/check
  internal/cnnhead/             голова CNN
  internal/cbfast/              CatBoost, резерв из go-inference-stand
  internal/features/            календарные признаки
  web/                          встроенный UI
frontend/                    5. будущий React-дашборд (пока README-заглушка)
data/                        всё, что читается кодом
  raw/                          train.csv, test.csv - gitignore
  labels/                       почасовая разметка - gitignore
  reference/calendar/           2024-2026.xml, единственная копия
  reference/spravochniki/
  external/weather|traffic|events|geo|factors/   собранные CSV и кэш OSM
artifacts/                   gitignore, кроме маленьких json-метаданных
deploy/Dockerfile, docker-compose.yml
tools/charts.py              графики для docs
tests/                       python-тесты по пакетам + общие фикстуры
Makefile                     единые точки входа
```

## Таблица переносов

| Сейчас | Куда | Что поменять |
|---|---|---|
| `data_preparation.py` | `tram_ml/features.py`; загрузка разметки - `tram_data/labels.py` | разделить функции |
| `metrics.py`, `backtest.py`, `harmonic.py`, `mlp.py` | `tram_ml/...` | импорты на `tram_ml.*` |
| `weekday_calibration.py` | `tram_ml/calibration.py` | объединить с калибровкой из `backtest` |
| `postprocess.py`, `final_ensemble.py` | `tram_ml/postprocess.py`, `tram_ml/ensemble.py` + `cli.py` | сделать `--weather all` дефолтом (открытое решение из MODEL.md) |
| `blend.py`, `twostage.py` | `research/archive/` | отклонённые подходы, в продакшн не нужны |
| `input/calendar/*` | `data/reference/calendar/` + `tram_data/calendar.py` | CNN переходит на тот же загрузчик, дубли удаляются |
| `weather/`, `traffic/`, `events/`, `factors/`, `geo/build_*` | код -> `tram_data/sources/`, данные -> `data/external/`, README -> `docs/notes/sources/` | пути через `paths.py` |
| `*/test_*.py` | `tests/tram_data/` | |
| `experiments/`, `geo/loro_experiment.py` | `research/horizon/`, `research/geo/` | |
| `*.ipynb` в корне | `research/notebooks/` | |
| `feature/cnn`: `src/`, `tests/`, `configs/` | `src/tram_forecast/`, `tests/tram_forecast/`, `configs/` | через `git mv`, чтобы сохранить историю |
| `feature/cnn`: `service/` + `go-inference-stand`: `bench/go-inference-stand/` | один модуль `service/` | CatBoost - альтернативный бэкенд модели за общим интерфейсом |
| `Dockerfile`, `docker-compose.yml` | `deploy/` | восстановить команду `export-head`, которой сейчас нет в CLI CNN |
| `artifacts/submissions/*.csv`, `artifacts/preds/*.parquet` | GitHub Release или Git LFS | `git rm --cached`; таблица `RESULTS.md` -> в docs |
| `catboost_info/`, `__pycache__/`, `.DS_Store` | удалить | уже в gitignore, но физически лежат |
| `final/` | содержимое -> корневой `README.md` и `docs/` | поправить путь в `tools/charts.py` |

## Сквозные правила

1. **Пути к данным** - только через `tram_data/paths.py`. В коде не остаётся ни одного
   `"dataset/..."`.
2. **Запуск только как модулей**: `python -m tram_ml ...`, `python -m tram_forecast ...`. Никаких
   `sys.path`-хаков (сейчас `# noqa: E402` в `geo/loro_experiment.py` ровно из-за этого).
3. **Единые точки входа в Makefile**:

   | Цель | Что делает |
   |---|---|
   | `data` | сбор внешних источников и почасовая сетка |
   | `train-ml` | сабмит ансамбля |
   | `train-cnn` | обучение CNN и сабмит |
   | `export` | голова CNN для Go |
   | `service` | запуск сервиса |
   | `load` | нагрузочный тест |
   | `figures` | графики для docs |
   | `test` | все тесты |

4. **Производные датасеты** собираются шагом пайплайна, а не копируются руками (например,
   `parking_events_hourly.csv` в CNN-ветке).
5. **Разметку организаторов не коммитить.** В `feature/cnn` закоммичен `dataset/labels` (около
   58 тыс. строк) - убрать и описать в README, откуда брать данные.

## Порядок работ

| Шаг | Что делаем | Критерий «ничего не сломали» |
|---|---|---|
| 0 | тег текущего состояния `main` и обеих веток | теги есть на remote |
| 1 | ветка `restructure`: уборка мусора, `pyproject.toml`, `paths.py` | тесты сборщиков зелёные |
| 2 | перенос ML в `src/tram_ml` | `python -m tram_ml build-submission --weather all` даёт файл, побайтно совпадающий с сабмитом 0.88466 |
| 3 | сборщики и данные в `tram_data` и `data/` | пересборка `factors_hourly` и `events_hourly` совпадает с текущими CSV |
| 4 | вливание `feature/cnn` через `git mv` | тесты CNN зелёные, прогноз тем же чекпойнтом совпадает |
| 5 | общий календарь для обеих моделей | признаки CNN до и после совпадают |
| 6 | слияние двух Go-сервисов в `service/` | `cmd/check` (golden-сверка) проходит для CatBoost и CNN |
| 7 | `deploy/`, сборка образа | `docker compose up` в чистом окружении, `/healthz` = ok |
| 8 | вынос бинарников из git, `final/` -> корень, `docs/notes/` | все ссылки в docs живые |
| 9 | CI: ruff + pytest + `go test` + сборка образа | зелёный прогон |

Шаги 2 и 4 независимы и могут идти параллельно; всё остальное - после шага 1 по порядку.

## Открытые решения

1. **Когда выполнять.** Дедлайн загрузки решения - 27 сентября, 23:59. В сданных материалах
   фигурируют команды и пути текущей структуры, поэтому до дедлайна имеет смысл сделать только
   перенос `final/` в корневые `README.md` и `docs/`, остальное - в ветке после.
2. **Бинарники: GitHub Release или Git LFS.** Чистка истории (`filter-repo`) не планируется:
   она необратима и ломает клоны команды. Достаточно перестать отслеживать файлы дальше.
3. **Один `pyproject.toml` с тремя пакетами** (как в плане) или workspace из трёх проектов.
   Предпочтение - один: проще, зависимости почти общие.
4. **CatBoost в сервисе** - второй бэкенд за флагом или только отдельный стенд для резерва.
