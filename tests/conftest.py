import pytest
import torch

torch.set_num_threads(1)

from tram_forecast.config import DataSettings, Settings
from tram_forecast.data import prepare


@pytest.fixture
def prepared(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "route;date;hour;boardings\n1;2025-01-01;0;2\n1;2025-01-22;0;3\n1;2025-02-01;0;10\n"
    )
    data = DataSettings(
        label_paths=(str(labels),),
        routes=(1,),
        start="2025-01-01",
        validation_cutoff="2025-02-01",
        final_cutoff="2025-03-01",
        forecast_end="2025-04-01",
        output_root=str(tmp_path / "prepared"),
        min_free_disk_bytes=0,
    )
    settings = Settings(data=data)
    path = prepare(settings, "validation")
    return settings, path
