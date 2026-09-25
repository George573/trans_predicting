import csv

import pytest

from tram_forecast.config import Config
from tram_forecast.preprocess import prepare
from tram_forecast.settings import DataSettings, Settings


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw.csv"
    labels = tmp_path / "labels.csv"
    with raw.open("w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "ngpt_route",
                "tran_date_time",
                "validation_result",
                "tran_type_id",
                "good_type",
                "place_id",
                "pass_route",
            ]
        )
        writer.writerows(
            [
                ["1 трамвай", "2025-01-21 23:59:00", "1", "52", "fare", "1", ""],
                ["1", "2025-01-01 00:00:00", "1", "52", "fare", "1", "0"],
                ["1", "2025-01-21 23:59:00", "1", "52", "second", "1", "nan"],
                ["1", "2025-02-01 00:00:00", "1", "52", "HELDOUT", "1", "new"],
            ]
        )
    labels.write_text(
        "route;date;hour;boardings\n1;2025-01-01;0;2\n1;2025-01-22;0;3\n1;2025-02-01;0;10\n"
    )
    data = DataSettings(
        raw_paths=(str(raw),),
        label_paths=(str(labels),),
        routes=(1,),
        start="2025-01-01",
        validation_cutoff="2025-02-01",
        final_cutoff="2025-03-01",
        forecast_end="2025-04-01",
        output_root=str(tmp_path / "prepared"),
        temp_dir=str(tmp_path / "tmp"),
        min_free_disk_bytes=0,
        fetch_rows=2,
    )
    settings = Settings(model=Config(min_frequency=1), data=data)
    path = prepare(settings, "validation")
    return settings, path
