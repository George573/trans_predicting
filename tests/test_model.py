import pytest
import torch

from tram_forecast.config import Config
from tram_forecast.model import ForecastNetwork

torch.set_num_threads(1)


def fixture():
    history = {
        "counts": torch.rand(1, 1, 504) * 50,
        "calendar": torch.rand(1, 6, 504),
    }
    request = {
        "route_indices": torch.tensor([1]),
        "calendar": torch.rand(1, 4),
        "lead": torch.tensor([61.0]),
    }
    return history, request


def test_network_parameters_forward_backward_and_cache():
    torch.manual_seed(1)
    model = ForecastNetwork(20).eval()
    assert model.parameter_report()["total"] == 195868
    assert set(model.parameter_report()) == {"boarding", "shared", "route", "head", "total"}
    history, request = fixture()
    out = model(history, request)
    assert out.shape == (1, 24) and torch.isfinite(out).all() and (out >= 0).all()
    torch.testing.assert_close(
        out,
        model.predict_day(
            model.encode_history(history),
            request["route_indices"],
            request["calendar"],
            request["lead"],
        ),
    )
    out.mean().backward()
    for module in [model.boarding, model.shared, model.route, model.head]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert torch.isfinite(model(history, request)).all()
    with pytest.raises(ValueError, match="unsupported"):
        model(history, dict(request, route_indices=torch.tensor([0])))


def test_state_dict_reload(tmp_path):
    torch.manual_seed(3)
    model = ForecastNetwork(1, Config(dropout=0)).eval()
    history, request = fixture()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = ForecastNetwork(1, Config(dropout=0)).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(history, request), restored(history, request))
