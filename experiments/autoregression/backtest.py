"""Repeat fixed seven-epoch direct-model training at an earlier forecast origin."""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch_transformer"))
from probe import detailed
from run import START, fit, load_data

from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.predict import fixed_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2025-07-02")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/autoregression_july")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cutoff = date.fromisoformat(args.cutoff)
    training = load_data(cutoff)
    validation = load_data(cutoff + timedelta(days=61))
    actual = validation.counts[:, (cutoff - START).days * 24 :].reshape(9, 61, 24)
    settings = SimpleNamespace(
        history_days=28, device="cpu", leads=12, batch_size=32, lr=0.001, patience=5
    )
    forecasts, results = {}, {}
    for kind, seed in [("cnn", 67), ("patch", 123)]:
        model, _ = fit(
            kind,
            training,
            settings,
            epochs=7,
            seed=seed,
            output=args.output / f"{kind}.pt",
        )
        forecasts[kind + "_direct"] = fixed_forecast(
            model, validation, cutoff, 61, getattr(model, "history_days", 14)
        )
        for block in (1, 7):
            forecasts[f"{kind}_recursive_{block}"] = recursive_forecast(
                model, validation, cutoff, block_days=block
            )
    forecasts["previous_blend"] = (
        0.75 * forecasts["patch_direct"] + 0.25 * forecasts["cnn_direct"]
    )
    for w in (0.25, 0.5, 0.75):
        forecasts[f"hybrid_{w}"] = (
            w * forecasts["patch_direct"] + (1 - w) * forecasts["cnn_recursive_1"]
        )
    for name, prediction in forecasts.items():
        results[name] = detailed(actual, prediction)
        print(name, results[name]["score"], flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", actual=actual, **forecasts)


if __name__ == "__main__":
    main()
