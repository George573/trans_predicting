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


# Школьные каникулы Москвы. Единого графика по городу нет: школы утверждают его сами,
# поэтому берём даты, на которых сходятся официальные приказы школ и рекомендации.
# Источник: ГБОУ Школа № 1517, приказ 92/ОРГ от 20.03.2025,
# https://1517.mskobr.ru/edu-news/11020
#
# Замер по данным (рабочие дни, отношение к соседним неделям того же дня недели):
# осенние каникулы 25-31 октября дают 0.985 против 1.017 на учебных неделях, то есть
# провал около 3%. Гипотеза про модульный перерыв 4-12 октября в данных не подтвердилась
# (1.006, провала нет), поэтому вариант с каникулами 15-23 ноября мы не размечаем.
#
# Для ноября-декабря 2025 признак почти инертен: внутри прогнозного окна лежат только
# зимние каникулы с 31 декабря, а этот день производственный календарь уже размечает
# как выходной. Признак нужен модели на других горизонтах, а не ради этого сабмита.
SCHOOL_VACATIONS: list[tuple[date, date]] = [
    (date(2025, 1, 1), date(2025, 1, 8)),      # зимние 2024/2025
    (date(2025, 3, 24), date(2025, 3, 30)),    # весенние 2024/2025
    (date(2025, 5, 26), date(2025, 8, 31)),    # летние
    (date(2025, 10, 25), date(2025, 11, 2)),   # осенние 2025/2026
    (date(2025, 12, 31), date(2026, 1, 11)),   # зимние 2025/2026
]

# Горизонт, на котором признак расстояния до праздничного блока вообще что-то значит.
# Дальше эффект в данных не отличим от нуля (контроль: 1.002 при |dist| > 5).
BLOCK_HORIZON = 7

#   набор                   фолд 1 (есть блоки)   фолд 3    фолд 4
#   cyclic                        0.1997           0.1595    0.1104
#   cyclic + блоки                0.1704           0.1519    0.1097
#   cyclic + блоки + каникулы     0.1704           0.1572    0.1147
#
# Блоки берём, флаг каникул - нет: он почти дублирует season (всё лето это каникулы) и на
# октябрьском окне портит метрику на 0.43 пункта. Оставлен отдельной константой, потому
# что на других горизонтах (дашборд, весна) может пригодиться.
CALENDAR_BLOCK_FEATURES: list[str] = ["days_before_block", "days_after_block", "in_block"]
SCHOOL_VACATION_FEATURES: list[str] = ["is_school_vacation"]


def holiday_blocks(calendar_df: pl.DataFrame, min_len: int = 3) -> list[tuple[date, date]]:
    """Подряд идущие нерабочие дни длиной не меньше min_len.

    Это и есть праздничные блоки: новогодний, майские, июньский, ноябрьский. Считаются
    из производственного календаря, а не задаются руками, поэтому переносятся на любой год.
    """
    off = set(
        calendar_df.filter(pl.col("is_weekend"))["date"].to_list()
    )
    # Новогодний блок переходит через границу года, а календарь у нас на один год. Без
    # этого 29-31 декабря не видят блока, к которому относятся. 1-8 января нерабочие
    # каждый год по статье 112 ТК РФ, поэтому дни следующего января можно добавить как
    # известный факт, а не как допущение.
    next_year = max(off).year + 1 if off else None
    if next_year is not None:
        off |= {date(next_year, 1, day) for day in range(1, 9)}
    blocks: list[tuple[date, date]] = []
    run_start: date | None = None
    prev: date | None = None
    for d in sorted(off):
        if run_start is None:
            run_start = d
        elif prev is not None and (d - prev).days > 1:
            if (prev - run_start).days + 1 >= min_len:
                blocks.append((run_start, prev))
            run_start = d
        prev = d
    if run_start is not None and prev is not None and (prev - run_start).days + 1 >= min_len:
        blocks.append((run_start, prev))
    return blocks


def add_block_features(calendar_df: pl.DataFrame) -> pl.DataFrame:
    """Разметка по дате: сколько дней до праздничного блока, сколько после, внутри ли он.

    Зачем не один знаковый признак: деревьям удобнее два неотрицательных, каждый со своим
    порогом, потому что спад перед блоком и после него разной глубины - 0.98 против 0.95
    по замеру на 2025 годе.

    Для нас это способ убрать ручные новогодние множители из postprocess: 29-30 декабря -
    это дни -2 и -1 перед зимним блоком, и такие же дни есть в обучении перед февральским,
    майскими и июньским блоками.
    """
    blocks = holiday_blocks(calendar_df)
    vac = SCHOOL_VACATIONS

    rows = []
    for d in calendar_df["date"].to_list():
        before, after, inside = BLOCK_HORIZON, BLOCK_HORIZON, False
        for lo, hi in blocks:
            if lo <= d <= hi:
                inside = True
                before = after = 0
                break
            if d < lo:
                before = min(before, (lo - d).days)
            else:
                after = min(after, (d - hi).days)
        rows.append({
            "date": d,
            "days_before_block": min(before, BLOCK_HORIZON),
            "days_after_block": min(after, BLOCK_HORIZON),
            "in_block": inside,
            "is_school_vacation": any(lo <= d <= hi for lo, hi in vac),
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("days_before_block").cast(pl.Int8),
        pl.col("days_after_block").cast(pl.Int8),
    )


# Внешние признаки из events/data/events_hourly.csv (см. events/README.md): ключ
# route x date x hour, покрытие 2025-2026, у каждого источника есть ссылка.
#
# route_change - изменения маршрутов по постам Дептранса: отмена, укорачивание, объезд.
#
# ВНИМАНИЕ: в модель НЕ включён, хотя на фолдах выглядит лучшей внешней фичей за всё время
# (пул фолдов 3+4: 0.1086 без неё против 0.0996 с ней, то есть 0.90 пункта). На реальной
# цели она сделала хуже: 0.88268 -> 0.88146.
#
# Почему фолды соврали. Все источники в route_changes.csv - посты Дептранса не позднее
# конца октября 2025, то есть сбор закончился там же, где обучающие данные. Маршруты 7 и 50
# стояли под ограничениями каждый день с середины августа (28, 30, 31 день в авг/сен/окт),
# записи доходят до 24 ноября, а в декабре их нет вообще. Модель с этой фичей списывает
# просадку 7-го и 50-го на ремонт и возвращает им полный уровень с 25 ноября - то есть
# уровень, которого у них не было с июля. Модель без фичи считает придавленный уровень их
# нормальным, и по лидерборду это оказалось верно: либо работы продлили и это не собрано,
# либо спрос не восстановился сразу. Затухающий вес half-life 60 делает это допущение сам.
#
# Общий урок для бэктеста: асимметрию покрытия внешнего источника фолды увидеть не могут.
# В каждом фолде и обучение, и валидация лежат внутри плотно собранного периода, а ошибка
# возникает только за его краем. Любую внешнюю фичу, собранную до края обучающих данных,
# надо проверять на полноту в прогнозном окне отдельно от фолдов.
#
# Календарные блоки off_run_* и days_*_long_weekend перекрывают наши CALENDAR_BLOCK_FEATURES
# и измеримо чуть лучше (0.0996 против 0.1000). Наши оставлены как запас: они считаются из
# одного производственного календаря и работают на любом годе, а этот CSV - только 2025-2026.
#
# Событийные колонки (event_near_route, event_phase, event_dist_km) НЕ берём. Контроль
# перемешиванием: те же колонки, привязанные к случайным датам, дают 0.0965 против 0.0976
# у настоящих, то есть прирост шёл не от событий, а от лишней свободы для сплитов. Это
# совпадает с их собственным отчётом: футбол и концерты на эти маршруты почти не влияют.
EXTERNAL_CAT_FEATURES: list[str] = ["route_change"]
EXTERNAL_BLOCK_FEATURES: list[str] = [
    "off_run_len", "off_run_pos", "days_to_long_weekend", "days_after_long_weekend",
    "is_workday_before_long_weekend", "extended_night_service",
]
EXTERNAL_FEATURES: list[str] = EXTERNAL_CAT_FEATURES + EXTERNAL_BLOCK_FEATURES


# Погода и трафик из factors/data/factors_hourly.csv (см. factors/README.md): ключ
# route x date x hour, весь 2025 год, у каждого источника ссылка на первоисточник.
#
# Покрытие проверено отдельно, как требует урок route_change:
#   погода (все 7 колонок)          обучение 1.000, прогноз 1.000 - асимметрии нет
#   mo_vehicles, accidents_*        0.66 / 0.67 и 1.00 / 1.00 - симметрично
#   codd_score                      0.36 / 0.66 - обратная асимметрия
#   parking_pct, parking_pct_route  0.67 / 1.00 и 0.61 / 0.90 - обратная асимметрия
#   yandex_score, delay_post_age_h  0.026 / 0.008 и 0.004 / 0.008 - слишком разрежено
#
# Замер на пуле фолдов 3+4 (текущий набор даёт 0.1071):
#   + погода, все показатели          0.1117
#   + ПЕРЕМЕШАННАЯ погода             0.1102
#   + только прогнозная погода        0.1086
#   + погода + трафик                 0.1307
#   + только трафик                   0.1312
#
# Вывод. Эффект погоды на посадки реален и измерен в factors/README.md с доверительными
# интервалами: дождь от 0.3 мм/ч даёт -6...-10%, сильнее всего весной и летом по выходным.
# Но в ноябре-декабре его почти нет физически - за два месяца всего 63 дневных часа с
# осадками от 0.3 мм/ч, - а восемь коррелированных непрерывных колонок дают деревьям
# свободу запоминать отдельные дни. Перемешанная погода вредит почти так же, как настоящая,
# то есть ущерб идёт от сложности модели, а не от неверного сигнала.
#
# Трафик хуже на порядок из-за обратной асимметрии покрытия: модель учится в основном на
# пропусках, а в прогнозном окне идёт по ветке, под которую у неё мало обучающих данных.
#
# Поэтому в прогнозной модели на 61 день этого блока нет. Правильное место для погоды и
# трафика - корректирующий коэффициент в сервисе (критерий 2в): множитель применяется
# поверх готового прогноза на те дни, где прогноз погоды реально существует, и не трогает
# остальные. Так фича деградирует с горизонтом честно, а не тянет за собой всю модель.
WEATHER_FEATURES: list[str] = [
    "temperature_c", "precipitation_mm", "wind_speed_ms", "cloud_cover_pct",
    "humidity_pct", "forecast_temperature_c", "forecast_precipitation_mm",
]
WEATHER_FORECAST_ONLY: list[str] = ["forecast_temperature_c", "forecast_precipitation_mm"]
TRAFFIC_FEATURES: list[str] = [
    "mo_vehicles", "accidents_moscow", "accidents_near_route",
    "codd_score", "parking_pct", "parking_pct_route",
]


def load_weather_features(
    path: str = "factors/data/factors_hourly.csv",
    with_traffic: bool = False,
) -> pl.DataFrame:
    """Погода (и опционально трафик) по ключу route x date x hour.

    Добавляет temp_anom - отклонение температуры от климатической нормы, считанной как
    центрированное среднее дневной температуры за +-7 дней. По замеру в factors/README.md
    отклонение температуры влияет заметно только весной, но признак дешёвый.
    """
    cols = WEATHER_FEATURES + (TRAFFIC_FEATURES if with_traffic else [])
    fac = pl.read_csv(path, try_parse_dates=True).with_columns(
        pl.col("route").cast(pl.Int32), pl.col("hour").cast(pl.Int8)
    )
    norm = (
        fac.group_by("date").agg(pl.col("temperature_c").mean().alias("t_day")).sort("date")
        .with_columns(
            pl.col("t_day").rolling_mean(window_size=15, center=True, min_samples=3).alias("t_norm")
        )
    )
    return (
        fac.join(norm.select(["date", "t_norm"]), on="date", how="left")
        .with_columns((pl.col("temperature_c") - pl.col("t_norm")).alias("temp_anom"))
        .select(["route", "date", "hour"] + cols + ["temp_anom"])
    )


def load_event_features(
    path: str = "events/data/events_hourly.csv",
) -> pl.DataFrame:
    """Почасовые внешние признаки, готовые к join по route x date x hour."""
    return (
        pl.read_csv(path, try_parse_dates=True)
        .with_columns(
            pl.col("route").cast(pl.Int32),
            pl.col("hour").cast(pl.Int8),
            pl.col("route_change").fill_null("none"),
        )
        .select(["route", "date", "hour"] + EXTERNAL_FEATURES)
    )


def add_external_features(df: pl.DataFrame, events_df: pl.DataFrame) -> pl.DataFrame:
    out = df.join(events_df, on=["route", "date", "hour"], how="left", validate="1:1")
    missing = out["route_change"].null_count()
    if missing:
        raise ValueError(f"events_hourly не покрыл {missing} строк сетки")
    return out


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
        .join(
            add_block_features(calendar_df),
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
        .with_columns(daylight_expr())
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


# Признаки-мосты через невиданный участок календаря. Ноября и декабря в обучении нет, поэтому
# day_of_year и month через синусы лежат вне диапазона и вредят (см. CYCLIC_YEAR). А длина
# светового дня в ноябре-декабре 7-9 часов - это в точности диапазон января-февраля, которые
# в обучении есть. То же делает температура: ноябрьские +2 градуса модель видела в марте.
#
# Гипотеза подтверждена сабмитом: погодный блок дал 0.88268 -> 0.88466 при том, что на фолдах
# стоил 0.39 пункта. Фолды этого увидеть не могут - их валидация лежит в сезоне, который в
# обучении представлен, и там температура не добавляет ничего сверх season.
MOSCOW_LAT_RAD = math.radians(55.7558)
SEASON_BRIDGE_FEATURES: list[str] = ["daylight_h"]


def daylight_expr() -> pl.Expr:
    """Длина светового дня в Москве, часы. Аналитическая формула, внешних данных не нужно."""
    decl = (math.radians(23.44) * (2 * math.pi * (284 + pl.col("day_of_year")) / 365.0).sin())
    cos_omega = (-math.tan(MOSCOW_LAT_RAD) * decl.tan()).clip(-1.0, 1.0)
    return (2 * cos_omega.arccos().degrees() / 15.0).alias("daylight_h")


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
