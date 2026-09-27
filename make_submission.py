"""Сборка файла прогноза классической модели одной командой.

    python3 make_submission.py                 # конфигурация лучшего сабмита, WAPE-score 0.88466
    python3 make_submission.py --no-weather    # только данные, известные на 01.11.2025, 0.88268

Нужны почасовые метки организаторов в dataset/labels/ (labels_day_train.csv, labels_day_test.csv)
и шаблон dataset/test_submission.csv. Всё остальное лежит в репозитории. Скрипт обучает ансамбль
CatBoost + MLP тем же кодом, что и отправленный сабмит (final_ensemble.py), выбирает вариант с
групповой поправкой и новогодним блоком, проверяет сетку и пишет submission.csv. Около 4-10 минут.
Попутно пайплайн обновляет промежуточные прогнозы в artifacts/preds/ и кладёт все четыре варианта
сборки в artifacts/submissions/.

Погодный блок - фактическая погода ноября-декабря 2025 года и прогнозы ECMWF за сутки до срока.
Организаторы разрешили внешние данные, опубликованные после 31.10.2025, но в момент построения
прогноза их не было; --no-weather собирает прогноз без них.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "dataset/labels/labels_day_train.csv",
    "dataset/labels/labels_day_test.csv",
    "dataset/test_submission.csv",
    "input/calendar/2025.xml",
]
WEATHER = "factors/data/factors_hourly.csv"
REFERENCE = {
    "all": "artifacts/submissions/submission_ens-cal-ny-wx_09-27-2026_01:31:22.csv",
    "off": "artifacts/submissions/submission_ens-cal-ny_09-27-2026_00:31:23.csv",
}
ROUTES = {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
ROWS = 10 * 61 * 24


def fail(msg: str) -> None:
    print(f"ошибка: {msg}", file=sys.stderr)
    sys.exit(1)


def check_inputs(weather: str) -> None:
    missing = [p for p in REQUIRED + ([WEATHER] if weather == "all" else []) if not (ROOT / p).exists()]
    if missing:
        fail("нет входных файлов: " + ", ".join(missing))


def validate(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, separator=";", try_parse_dates=True)
    if df.columns != ["route", "date", "hour", "prediction"]:
        fail(f"колонки {df.columns}, нужны route;date;hour;prediction")
    template = pl.read_csv(ROOT / "dataset/test_submission.csv", separator=";", try_parse_dates=True)
    keys = ["route", "date", "hour"]
    checks = {
        f"{ROWS} строк": df.height == ROWS,
        "ключи уникальны": not df.select(keys).is_duplicated().any(),
        "маршруты 1, 5, 7, 11, 12, 17, 25, 26, 28, 50": set(df["route"].unique()) == ROUTES,
        "даты 2025-11-01 ... 2025-12-31": (str(df["date"].min()), str(df["date"].max())) == ("2025-11-01", "2025-12-31"),
        "часы 0-23": set(df["hour"].unique()) == set(range(24)),
        "прогноз целый и неотрицательный": df["prediction"].dtype.is_integer() and df["prediction"].min() >= 0,
        "ключи совпадают с шаблоном": df.select(keys).sort(keys).equals(template.select(keys).sort(keys)),
    }
    for name, ok in checks.items():
        print(f"  {'ok ' if ok else 'НЕТ'} {name}")
    if not all(checks.values()):
        fail("файл не прошёл проверку")
    return df


def compare(df: pl.DataFrame, weather: str) -> None:
    ref_path = ROOT / REFERENCE[weather]
    if not ref_path.exists():
        return
    ref = pl.read_csv(ref_path, separator=";", try_parse_dates=True)
    j = df.join(ref, on=["route", "date", "hour"], suffix="_ref")
    diff = (j["prediction"] - j["prediction_ref"]).abs()
    print(f"  против отправленного {ref_path.name}: расхождение {diff.sum() / j['prediction_ref'].sum():.4%} "
          f"от суммы, совпало строк {int((diff == 0).sum())} из {j.height}")
    print("  (MLP обучается на MPS или CPU недетерминированно, поэтому совпадение не побитовое)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Собирает submission.csv классической модели")
    ap.add_argument("--no-weather", action="store_true", help="без погоды прогнозного периода")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    args = ap.parse_args()
    weather = "off" if args.no_weather else "all"

    check_inputs(weather)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import final_ensemble

    tag = "-jury-wx" if weather == "all" else "-jury"
    before = set((ROOT / "artifacts/submissions").glob(f"submission_ens-cal-ny{tag}_*.csv"))
    print(f"обучение ансамбля, погода: {'включена' if weather == 'all' else 'выключена'}")
    final_ensemble.main(tag=tag, weather_mode=weather)
    made = sorted(set((ROOT / "artifacts/submissions").glob(f"submission_ens-cal-ny{tag}_*.csv")) - before)
    if not made:
        fail("пайплайн не записал файл прогноза")

    print(f"\nпроверка {made[-1].name}:")
    df = validate(made[-1])
    compare(df, weather)
    shutil.copyfile(made[-1], args.out)
    print(f"\nготово: {args.out} ({df.height} строк, сумма посадок {df['prediction'].sum():,})")


if __name__ == "__main__":
    main()
