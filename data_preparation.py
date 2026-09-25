import math

import polars as pl

from datetime import date

from input.calendar.transform import extract_holidays_from_local_file


def densify(
    df: pl.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    routes: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    start = start_date or df["date"].min()
    end = end_date or df["date"].max()
    route_values = routes or tuple(sorted(df["route"].unique().to_list()))

    grid = (
        pl.DataFrame({
            "date": pl.date_range(start, end, interval="1d", eager=True),
        })
        .join(
            pl.DataFrame({"hour": pl.Series(range(24), dtype=df.schema["hour"])}),
            how="cross",
        )
        .join(
            pl.DataFrame({"route": pl.Series(route_values, dtype=df.schema["route"])}),
            how="cross",
        )
    )

    return (
        grid
        .join(df, on=["date", "hour", "route"], how="left", validate="1:1")
        .with_columns(
            pl.col("target").fill_null(0).cast(pl.Int32)
        )
        .sort(["date", "hour", "route"])
    )


def load_calendar(calendar_path: str) -> pl.DataFrame:
    df = extract_holidays_from_local_file(calendar_path)
    return df


def load_dataset(
    data_path: str,
    start_date: date | None = None,
    end_date: date | None = None,
    dense: bool = True,
    routes: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    lf: pl.LazyFrame = pl.scan_csv(data_path, separator=';')
    df: pl.DataFrame = (
        lf
        .select(['tran_date_time', 'validation_result', 'ngpt_route'])
        .filter(pl.col('validation_result') == 1)
        .with_columns(
            pl.col('tran_date_time').str.to_datetime("%Y-%m-%d %H:%M:%S").alias('tran_date_time'),
            pl.col("ngpt_route").str.extract(r"(\d+)", group_index=1).cast(pl.Int32).alias("route")
        )
        .with_columns(
            pl.col('tran_date_time').dt.date().alias('date'),
            pl.col('tran_date_time').dt.hour().alias('hour')
        )
        .select(['date', 'hour', 'route'])
        .group_by(['date', 'hour', 'route'])
        .agg(pl.len().cast(pl.Int32).alias('target'))
        .collect(engine="streaming")
    )

    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)

    if dense:
        df = densify(df, start_date, end_date, routes)

    return df


def load_labels(
    labels_path: str,
    start_date: date | None = None,
    end_date: date | None = None,
    dense: bool = True,
    routes: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Готовая почасовая разметка из dataset/labels.

    Это те же агрегаты, что считает load_dataset из сырых CSV, но читаются за доли
    секунды вместо минут. Полезно для быстрых экспериментов; финальные прогоны
    имеет смысл сверять с load_dataset.
    """
    df = (
        pl.read_csv(labels_path, separator=';', try_parse_dates=True)
        .rename({"boardings": "target"})
        .select(["date", "hour", "route", "target"])
        .with_columns(
            pl.col("hour").cast(pl.Int8),
            pl.col("route").cast(pl.Int32),
            pl.col("target").cast(pl.Int32),
        )
    )

    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)

    if dense:
        df = densify(df, start_date, end_date, routes)

    return df


def enrich_features(
    base_df: pl.DataFrame,
    calendar_df: pl.DataFrame
) -> pl.DataFrame:
    enriched_df = (
        base_df
        .join(
            calendar_df,
            on="date",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("is_holiday").is_null())
            .then(False)
            .otherwise(pl.col("is_holiday"))
            .alias("is_holiday"),
            pl.when(pl.col("is_weekend").is_null())
            .then(False)
            .otherwise(pl.col("is_weekend"))
            .alias("is_weekend"),
            pl.when(pl.col("is_short_working_day").is_null())
            .then(False)
            .otherwise(pl.col("is_short_working_day"))
            .alias("is_short_working_day"),
        )
        .with_columns(
            (
                pl
                .when(pl.col("date").dt.month().is_in([12, 1, 2]))
                .then(pl.lit("winter"))
                .when(pl.col("date").dt.month().is_in([3, 4, 5]))
                .then(pl.lit("spring"))
                .when(pl.col("date").dt.month().is_in([6, 7, 8]))
                .then(pl.lit("summer"))
                .otherwise(pl.lit("autumn"))
                .alias("season")
            )
        )
        .with_columns(
            pl.col("date")
            .dt.strftime("%Y-%m-%d")
            .alias("date_str")
        )
        .with_columns(
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.day().alias("day_of_month"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
            pl.col("date").dt.month().alias("month"),
        )
        .with_columns(add_cyclic_features())
    )
    return enriched_df


# внутринедельные циклы: их значения в ноябре-декабре ровно те же, что в истории
CYCLIC_INTRAWEEK: list[str] = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "day_of_month_sin", "day_of_month_cos",
]

# годовые циклы: ноября и декабря в обучении нет, поэтому эти значения модель видит впервые
CYCLIC_YEAR: list[str] = [
    "day_of_year_sin", "day_of_year_cos",
    "month_sin", "month_cos",
]

CYCLIC_FEATURES: list[str] = CYCLIC_INTRAWEEK + CYCLIC_YEAR


def add_cyclic_features() -> list[pl.Expr]:
    """Синус-косинусное кодирование цикличных признаков.

    Обычный номер часа разрывает сутки между 23 и 0, номер дня недели - неделю между
    воскресеньем и понедельником. Пара sin/cos кладёт цикл на окружность, где эти точки
    соседние, и расстояние между значениями становится осмысленным.

    Для линейных моделей это даёт точную экстраполяцию: синус определён для любой даты,
    поэтому форма декабря считается по формуле. Деревьям пара sin/cos сама по себе
    экстраполировать не помогает - вне обученного диапазона они всё так же упираются в
    последний лист, - но разбиения становятся осмысленнее на границах цикла.

    Осторожно с day_of_year и month: ноябрь и декабрь в обучении не встречаются вообще,
    поэтому их sin/cos лежат в невиданной области. Ровно поэтому наборы фич с ними и без
    них перебираются подбором, а не берутся по умолчанию.
    """
    two_pi = 2 * math.pi
    day_of_month_period = pl.col("date").dt.month_end().dt.day().cast(pl.Float64)
    year_period = pl.when(pl.col("date").dt.is_leap_year()).then(366.0).otherwise(365.0)

    specs = [
        ("hour", pl.col("hour").cast(pl.Float64), pl.lit(24.0)),
        ("weekday", pl.col("weekday").cast(pl.Float64) - 1.0, pl.lit(7.0)),
        ("day_of_month", pl.col("day_of_month").cast(pl.Float64) - 1.0, day_of_month_period),
        ("day_of_year", pl.col("day_of_year").cast(pl.Float64) - 1.0, year_period),
        ("month", pl.col("month").cast(pl.Float64) - 1.0, pl.lit(12.0)),
    ]

    exprs: list[pl.Expr] = []
    for name, value, period in specs:
        angle = two_pi * value / period
        exprs.append(angle.sin().alias(f"{name}_sin"))
        exprs.append(angle.cos().alias(f"{name}_cos"))
    return exprs

def build_submit(
    start_date: date = date(2025, 11, 1),
    end_date: date = date(2025, 12, 31),
    routes: tuple[int, ...] = (12, 5, 1, 28, 50, 25, 11, 7, 26, 17),
) -> pl.DataFrame:
    dates_df = pl.DataFrame({
        "date": pl.date_range(
            start=start_date,
            end=end_date,
            interval="1d",
            eager=True,
        )
    })

    hours_df = pl.DataFrame({
        "hour": pl.Series(range(24), dtype=pl.Int8)
    })

    routes_df = pl.DataFrame({
        "route": pl.Series(routes, dtype=pl.Int32)
    })

    return (
        dates_df
        .join(hours_df, how="cross")
        .join(routes_df, how="cross")
        .select([
            "date",
            "hour",
            "route",
        ])
        .sort(["date", "hour", "route"])
    )
