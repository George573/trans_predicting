"""Notebook wiring checks; training is replaced with an untrained checkpoint."""

import json
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/train.ipynb"


def test_notebook_is_clean_and_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")


@pytest.mark.parametrize("kind", ["boarding_only"])
def test_notebook_fixture_workflow(prepared, tmp_path, monkeypatch, kind):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("IPython")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from tram_forecast.checkpoint import save_checkpoint
    from tram_forecast.evaluate import evaluate_model
    from tram_forecast.io import write_json
    from tram_forecast.model import ForecastNetwork
    from tram_forecast.storage import Store

    settings, artifact = prepared
    calls = []

    def fixture_train(settings, artifact, model_kind, output, resume):
        calls.append((model_kind, resume))
        store = Store(artifact)
        model = ForecastNetwork(
            store.metadata["scale"],
            settings.model, model_kind,
        )
        report = evaluate_model(model, store, settings.training.forecast_days)
        checkpoint = output / "best.pt"
        save_checkpoint(checkpoint, model, store, settings)
        write_json(output / "history.json", [
            {"epoch": 1, "mae": 1.0, "validation": report},
        ])
        return checkpoint

    def no_optimizer(*args, **kwargs):
        raise AssertionError("Notebook verification must not optimize")

    monkeypatch.setattr("tram_forecast.train.train", fixture_train)
    monkeypatch.setattr(torch.optim.AdamW, "step", no_optimizer)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(NOTEBOOK.parent)
    namespace = {}
    notebook = json.loads(NOTEBOOK.read_text())
    try:
        for cell in notebook["cells"]:
            # Final refit/export is separately covered by execution and export tests.
            if cell["id"] == "final-forecast":
                break
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
            if cell["id"] == "configuration":
                namespace.update(
                    settings=settings, ARTIFACT=artifact,
                    RUN_DIR=tmp_path / "notebook_run", MODEL_KIND=kind,
                )
        assert calls == [(kind, None)]
        assert namespace["prediction"].shape == (1, 7, 24)
        assert namespace["report"]["forecast_days"] == 7
        assert (tmp_path / "notebook_run/evaluation.json").is_file()
    finally:
        plt.close("all")
