"""Смесь нескольких submission: взвешенное среднее прогнозов по ключу route/date/hour.

    python blend.py artifacts/submissions/a.csv artifacts/submissions/b.csv
    python blend.py a.csv b.csv --weights 0.6 0.4

Модели ошибаются в разных местах, поэтому смесь обычно точнее любой из них по
отдельности. Веса стоит держать грубыми (0.5 / 0.5), иначе это подгонка под лидерборд.
"""

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

KEY = ["route", "date", "hour"]
SUBMIT_DIR = Path("artifacts/submissions")


def read_submission(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, separator=";")
        .select(KEY + ["prediction"])
        .with_columns(pl.col("prediction").cast(pl.Float64))
        .sort(KEY)
    )


def blend(paths: list[str | Path], weights: list[float] | None = None) -> pl.DataFrame:
    frames = [read_submission(p) for p in paths]
    w = weights or [1.0 / len(frames)] * len(frames)
    if len(w) != len(frames):
        raise ValueError("число весов не совпадает с числом файлов")
    total_w = sum(w)

    merged = frames[0].rename({"prediction": "p0"})
    for i, f in enumerate(frames[1:], start=1):
        merged = merged.join(f.rename({"prediction": f"p{i}"}), on=KEY, how="inner", validate="1:1")
    if merged.height != frames[0].height:
        raise ValueError(f"ключи не совпали: {frames[0].height} против {merged.height}")

    mix = sum((pl.col(f"p{i}") * wi for i, wi in enumerate(w)), start=pl.lit(0.0)) / total_w
    return (
        merged
        .with_columns(mix.round(0).cast(pl.Int64).alias("prediction"))
        .select(KEY + ["prediction"])
        .sort(KEY)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    args = ap.parse_args()

    frames = [read_submission(p) for p in args.paths]
    for p, f in zip(args.paths, frames):
        print(f"{Path(p).name}: сумма {int(f['prediction'].sum()):,}")

    # насколько прогнозы расходятся между собой
    if len(frames) == 2:
        j = frames[0].join(frames[1], on=KEY, how="inner", suffix="_b")
        diff = (j["prediction"] - j["prediction_b"]).abs()
        level = (j["prediction"] + j["prediction_b"]) / 2
        print(f"среднее расхождение: {diff.sum() / level.sum():.1%} от уровня")

    out = blend(args.paths, args.weights)
    assert out.height == 10 * 61 * 24, f"ожидалось 14640 строк, получено {out.height}"
    assert out["prediction"].min() >= 0

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMIT_DIR / f"submission_blend_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}.csv"
    out.write_csv(path, separator=";")
    print(f"\nзаписано {out.height} строк в {path}")
    print(f"сумма смеси: {int(out['prediction'].sum()):,}")


if __name__ == "__main__":
    main()
