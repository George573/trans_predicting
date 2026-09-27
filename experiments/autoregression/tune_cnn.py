"""Select a long-horizon-trained CNN checkpoint using recursive 61-day WAPE."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from probe import detailed
from run import CUTOFF, FINAL, START, fit, load_data

from tram_forecast.autoregressive import recursive_forecast


def recursive_validation(model, data, cutoff, days, history_days):
    return recursive_forecast(model, data, cutoff, days, block_days=1)


def main():
    torch.set_num_threads(4)
    out = Path("outputs/autoregression_cnn_selected")
    out.mkdir(exist_ok=False)
    args = SimpleNamespace(
        history_days=28, device="cpu", leads=12, batch_size=32, lr=0.001, patience=5
    )
    training, validation = load_data(CUTOFF), load_data(FINAL)
    model, epoch = fit(
        "cnn",
        training,
        args,
        16,
        validation=validation,
        seed=67,
        output=out / "cnn.pt",
        validation_forecast=recursive_validation,
    )
    actual = validation.counts[:, (CUTOFF - START).days * 24 :].reshape(9, 61, 24)
    predicted = recursive_forecast(model, validation, CUTOFF)
    result = dict(**detailed(actual, predicted), epoch=epoch)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "predictions.npz", actual=actual, cnn=predicted)


if __name__ == "__main__":
    main()
