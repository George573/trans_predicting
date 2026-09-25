from copy import deepcopy

import torch

from tram_forecast.collate import pack_hours
from tram_forecast.config import Config
from tram_forecast.model import ForecastNetwork
from tram_forecast.model.events import EventEncoder


def test_chunk_tiling_checkpoint_equivalence():
    torch.manual_seed(12)
    config = Config(max_hours=2, max_positions=28)
    encoder = EventEncoder([9] * 5, config).train()
    reference = deepcopy(encoder)
    hours = [torch.randint(1, 9, (n, 5)) for n in (0, 1, 2, 19, 71, 3, 51)]
    chunks = list(pack_hours(hours, 2, 28))
    assert any(c.tiled for c in chunks)
    for c in chunks:
        if not c.tiled:
            assert len(c.indices) * max(len(hours[i]) for i in c.indices) <= 28
    actual = encoder.encode_hours(hours)
    expected = torch.stack(
        [
            (
                reference(h.unsqueeze(0), torch.tensor([len(h)]))[0]
                if len(h)
                else torch.zeros(32)
            )
            for h in hours
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for p, q in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)


def test_tile_ties_preserve_first_gradient():
    torch.manual_seed(4)
    encoder = EventEncoder([5] * 5, Config(max_positions=20)).train()
    # Constant interior features create tied maxima across many cores.
    for block in (encoder.e1, encoder.e2):
        for path in block.paths:
            torch.nn.init.constant_(path.weight, 0.02)
            torch.nn.init.constant_(path.bias, 0.1)
    for emb in encoder.embeddings:
        torch.nn.init.constant_(emb.weight, 0.1)
    reference = deepcopy(encoder)
    hour = torch.ones(81, 5, dtype=torch.long)
    actual = encoder.encode_hours([hour])
    expected = reference(hour[None], torch.tensor([81]))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for p, q in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)


def test_ragged_network_matches_reference():
    model = ForecastNetwork([9] * 5, 1, Config(max_positions=28)).eval()
    hours = [torch.empty(0, 5, dtype=torch.long) for _ in range(504)]
    hours[7] = torch.randint(1, 9, (60, 5))
    counts = torch.zeros(1, 1, 504)
    calendar = torch.rand(1, 6, 504)
    a = model.encode_history({"hours": hours, "counts": counts, "calendar": calendar})
    b = model.encode_history(
        {
            "event_ids": hours[7][None],
            "lengths": torch.tensor([60]),
            "hour_indices": torch.tensor([7]),
            "counts": counts,
            "calendar": calendar,
        }
    )
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    empty = model.events.encode_hours([torch.empty(0, 5, dtype=torch.long)] * 4)
    assert torch.equal(empty, torch.zeros(4, 32))
