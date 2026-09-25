"""Постобработка готового submission: поправки, которые не требуют переобучения.

Каждая поправка - явный корректирующий коэффициент с обоснованием, а не подгонка:

1. Маршрут 5. В истории янв-окт по нему ноль успешных валидаций, но организаторы
   сообщили, что в эталоне ноя-дек он присутствует и нули по нему стоят около 0.0065
   метрики (максимум с нулями - 0.9935). Заполняем его городским профилем,
   отмасштабированным на эту долю.

2. Новогодний блок. 31 декабря 2025 - официальный выходной (перенос с 1 мая,
   в производственном календаре d="12.31" t="1" f="01.05"), и пайплайн это уже учитывает:
   модели прогнозируют его на уровне выходного дня, 122 тыс. против 243 тыс. в среду.
   Поэтому поправка здесь маленькая и только на вечер - большая часть падения уже учтена
   календарём, а двойной счёт утопил бы день ниже любого разумного уровня.

   Что говорят данные: в январе праздничный блок 2-8 числа идёт на 105-118 тыс. в день,
   то есть примерно как обычный январский выходной (114 тыс.), и только само 1 января
   проваливается вдвое (54.8 тыс.). Первые рабочие дни после каникул, 9-10 января, дают
   0.91-0.93 от нормы - по симметрии предпраздничные 29-30 декабря взяты так же.

   Вечер 31 декабря занижен умеренно: после 18 часов поездок мало, но НГПТ в Москве
   работает всю новогоднюю ночь, поэтому час 23 не трогаем.

3. Ночное окно. Трамвай не ходит примерно с 01:00 до 05:30. Поправка оставлена
   параметром, но по умолчанию выключена: модели, обученные на плотной сетке, уже
   кладут в часы 1-4 те же 0.04% объёма, что и в истории, и зануление ничего не даёт
   (замер на фолдах: 0.00 пункта). Включать имеет смысл только для моделей, обученных
   на разреженной сетке.
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import polars as pl

SUBMIT_DIR = Path("artifacts/submissions")

ROUTE5_SHARE = 0.0065          # доля маршрута 5 в эталоне, со слов организаторов
NIGHT_HOURS = (1, 2, 3, 4)     # 01:00-05:00; час 5 не трогаем, движение начинается в 05:30

# множители на дату: доля от того, что выдала модель
NEW_YEAR_DAYS: dict[date, float] = {
    date(2025, 12, 29): 0.95,
    date(2025, 12, 30): 0.92,
    date(2025, 12, 31): 0.92,   # день уже учтён календарём как выходной
}
# форма суток 31 декабря: вечер проседает сильнее утра. Это множители ПОВЕРХ дневного,
# в среднем по объёму около единицы, чтобы дневной коэффициент оставался управляющим.
NEW_YEAR_HOURLY: dict[date, dict[range, float]] = {
    date(2025, 12, 31): {range(0, 12): 1.05, range(12, 18): 1.00, range(18, 23): 0.80, range(23, 24): 1.00},
}


def read_submission(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, separator=";")
        .with_columns(pl.col("prediction").cast(pl.Float64))
        .sort(["route", "date", "hour"])
    )


def fill_route5(df: pl.DataFrame, share: float = ROUTE5_SHARE) -> pl.DataFrame:
    """маршрут 5 получает городской профиль, отмасштабированный на долю share"""
    others = df.filter(pl.col("route") != 5)
    total = others["prediction"].sum()
    if total <= 0:
        return df

    target_total = total * share / (1.0 - share)
    city = (
        others.group_by(["date", "hour"]).agg(pl.col("prediction").sum().alias("w"))
        .with_columns((pl.col("w") / pl.col("w").sum() * target_total).alias("prediction"))
        .select(["date", "hour", "prediction"])
        .with_columns(pl.lit(5, dtype=df.schema["route"]).alias("route"))
    )
    return pl.concat([others, city.select(df.columns)], how="vertical").sort(["route", "date", "hour"])


def adjust_new_year(
    df: pl.DataFrame,
    days: dict[date, float] = NEW_YEAR_DAYS,
    hourly: dict[date, dict[range, float]] = NEW_YEAR_HOURLY,
) -> pl.DataFrame:
    factor = pl.lit(1.0)
    for d, k in days.items():
        factor = pl.when(pl.col("date") == d.isoformat()).then(pl.lit(k)).otherwise(factor)
    shape = pl.lit(1.0)
    for d, spans in hourly.items():
        for hours, k in spans.items():
            shape = (
                pl.when((pl.col("date") == d.isoformat()) & pl.col("hour").is_in(list(hours)))
                .then(pl.lit(k))
                .otherwise(shape)
            )
    return df.with_columns((pl.col("prediction") * factor * shape).alias("prediction"))


def zero_night(df: pl.DataFrame, hours: tuple[int, ...] = NIGHT_HOURS) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("hour").is_in(list(hours))).then(pl.lit(0.0)).otherwise(pl.col("prediction")).alias("prediction")
    )


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    out = (
        df.with_columns(pl.col("prediction").clip(0).round(0).cast(pl.Int64))
        .select(["route", "date", "hour", "prediction"])
        .sort(["route", "date", "hour"])
    )
    assert out.height == 14640, f"ожидалось 14640 строк, получено {out.height}"
    assert out["date"].n_unique() == 61
    assert out["route"].n_unique() == 10
    assert out["prediction"].min() >= 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--route5", type=float, default=ROUTE5_SHARE, help="доля маршрута 5, 0 - оставить нули")
    ap.add_argument("--no-new-year", action="store_true")
    ap.add_argument("--night", action="store_true", help="занулить часы 01-05")
    ap.add_argument("--tag", default="adj")
    ap.add_argument("--out", default=None, help="явное имя файла в artifacts/submissions")
    args = ap.parse_args()

    df = read_submission(args.path)
    before = df["prediction"].sum()
    print(f"исходный файл: {df.height} строк, сумма {int(before):,}")

    if args.route5 > 0:
        df = fill_route5(df, args.route5)
        r5 = df.filter(pl.col("route") == 5)["prediction"].sum()
        print(f"маршрут 5 заполнен: {int(r5):,} посадок ({r5 / df['prediction'].sum() * 100:.2f}% итога)")
    if not args.no_new_year:
        df = adjust_new_year(df)
        print(f"новогодний блок: {', '.join(f'{d} x{k}' for d, k in NEW_YEAR_DAYS.items())}, "
              f"31 декабря дополнительно по часам")
    if args.night:
        df = zero_night(df)
        print(f"ночные часы {NIGHT_HOURS} занулены")

    out = finalize(df)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        path = SUBMIT_DIR / args.out
    else:
        path = SUBMIT_DIR / f"submission_{args.tag}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}.csv"
    out.write_csv(path, separator=";")
    print(f"\nитоговая сумма {int(out['prediction'].sum()):,} "
          f"({(out['prediction'].sum() / before - 1) * 100:+.2f}% к исходной)")
    print(f"записано в {path}")


if __name__ == "__main__":
    main()
