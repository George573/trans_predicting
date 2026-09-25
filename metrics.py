import numpy as np
import polars as pl


def wape(y_true, y_pred) -> float:
    """то же, что на лидерборде: одна глобальная дробь, а не среднее по группам"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    total = y.sum()
    return float(np.abs(y - p).sum() / total) if total > 0 else float("nan")


def wape_score(y_true, y_pred) -> float:
    return max(0.0, 1.0 - wape(y_true, y_pred))


def wape_report(y_true, y_pred, groups, group_name: str = "route") -> pl.DataFrame:
    """разбивка метрики по группам.

    wape_contrib - сколько пунктов ГЛОБАЛЬНОГО WAPE съедает группа, сумма по группам
    равна глобальному WAPE. Именно эта колонка говорит, где оптимизировать.
    wape_local - метрика внутри группы, для сравнения групп между собой.
    bias - системный сдвиг: > 0 перепрогноз, < 0 недобор.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    total = y.sum()

    return (
        pl.DataFrame({group_name: np.asarray(groups), "y": y, "p": p, "err": np.abs(y - p)})
        .group_by(group_name)
        .agg(
            pl.col("y").sum().alias("y_sum"),
            pl.col("p").sum().alias("p_sum"),
            pl.col("err").sum().alias("err_sum"),
        )
        .with_columns(
            pl.when(pl.col("y_sum") > 0)
            .then(pl.col("err_sum") / pl.col("y_sum"))
            .alias("wape_local"),
            (pl.col("err_sum") / total).alias("wape_contrib"),
            (pl.col("y_sum") / total).alias("volume_share"),
            pl.when(pl.col("y_sum") > 0)
            .then(pl.col("p_sum") / pl.col("y_sum") - 1)
            .alias("bias"),
        )
        .sort("wape_contrib", descending=True)
    )
