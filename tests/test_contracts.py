from datetime import date, datetime

import numpy as np
import pytest

from tram_forecast.config import ROUTES, Config
from tram_forecast.dataset import epoch_order, sample_index
from tram_forecast.schema import (
    calendar,
    count_scale,
    label_grid,
    parse_route,
    read_events,
    request_calendar,
)


def test_config_roundtrip_and_rejects_single_path(tmp_path):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(Config().to_dict()))
    assert Config.load(path).fingerprint == Config().fingerprint
    with pytest.raises(ValueError):
        Config(event1=((32, 3, 1),))
    with pytest.raises(ValueError):
        Config(max_positions=12)


def test_sample_boundaries_and_shuffle():
    rows = sample_index(ROUTES, date(2025, 1, 1), date(2025, 9, 1))
    assert len(rows) == 13797
    assert len(set(rows)) == len(rows)
    assert min(x.cutoff for x in rows) == date(2025, 1, 22)
    assert max(x.requested for x in rows) == date(2025, 8, 31)
    assert {x.lead for x in rows} == set(range(1, 8))
    assert sum(x.lead == 7 for x in rows) == 216 * 9
    extended = sample_index(ROUTES, date(2025, 1, 1), date(2025, 9, 1), 61)
    assert len(extended) == 105408
    order = epoch_order(len(rows), 0)
    assert len(set(order)) == len(rows)
    assert np.array_equal(order, epoch_order(len(rows), 0))
    assert not np.array_equal(order, epoch_order(len(rows), 1))


def test_labels_calendar_and_heldout_scale(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("route;date;hour;boardings\n1;2025-01-01;0;48\n1;2025-02-01;0;9999\n")
    y, mask = label_grid([p], [1], date(2025, 1, 1), date(2025, 1, 2))
    assert y.sum() == 48 and mask.sum() == 1 and count_scale(y) == 2
    c = calendar([datetime(2025, 9, 1)])
    np.testing.assert_allclose(c[:, 0], [0, 1, 0, 1, 0, 1], atol=1e-7)
    np.testing.assert_array_equal(request_calendar([date(2025, 9, 1)]), c[2:].T)
    p.write_text(p.read_text() + "1;2025-01-01;0;48\n")
    with pytest.raises(ValueError, match="duplicate"):
        label_grid([p], [1], date(2025, 1, 1), date(2025, 1, 2))


def test_event_parser_and_ties(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text(
        "ngpt_route;tran_date_time;validation_result;tran_type_id;good_type;place_id;pass_route\n1 трамвай;2025-01-01 01:00:00;1;52;A;1;\n1;2025-01-01 00:00:00;1;52;A;1;0\n1;2025-01-01 00:00:00;1;52;A;1;nan\n"
    )
    rows = sorted(read_events(p))
    assert [r[3] for r in rows] == [3, 4, 2]
    assert [r[4][-1] for r in rows] == ["0", "nan", None]
    assert parse_route(" 25 трамвай ") == 25
    with pytest.raises(ValueError):
        parse_route("25 bus")
