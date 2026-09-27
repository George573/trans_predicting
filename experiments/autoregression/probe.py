"""Compare direct checkpoints under recursive and anchored inference."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch_transformer"))
from run import CUTOFF, FINAL, START, load, load_data, metric

from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.predict import fixed_forecast


def detailed(actual, predicted):
    result = metric(actual, predicted)
    result["lead_scores"] = {
        f"{a + 1}-{b}": metric(actual[:, a:b], predicted[:, a:b])["score"]
        for a, b in ((0, 7), (7, 28), (28, 61))
    }
    return result


def main():
    torch.set_num_threads(4)
    out = Path("outputs/autoregression_probe")
    out.mkdir(exist_ok=False)
    data = load_data(FINAL)
    actual = data.counts[:, (CUTOFF - START).days * 24 :].reshape(9, 61, 24)
    results, predictions = {}, {}
    root = Path("outputs/patch_transformer_trial")
    for name in ("patch_123", "cnn_67"):
        model = load(root / f"{name}.pt")
        direct = fixed_forecast(
            model, data, CUTOFF, 61, getattr(model, "history_days", 14)
        )
        predictions[name + "_direct"] = direct
        for block in (1, 7):
            predictions[f"{name}_recursive_{block}"] = recursive_forecast(
                model, data, CUTOFF, block_days=block
            )
            predictions[f"{name}_anchored_{block}"] = recursive_forecast(
                model,
                data,
                CUTOFF,
                block_days=block,
                anchor=direct,
                recursive_weight=0.5,
            )
    for mode in ("direct", "recursive_1", "recursive_7", "anchored_1", "anchored_7"):
        predictions["blend_" + mode] = (
            0.75 * predictions["patch_123_" + mode]
            + 0.25 * predictions["cnn_67_" + mode]
        )
    for name, predicted in predictions.items():
        results[name] = detailed(actual, predicted)
        print(name, results[name]["score"], results[name]["lead_scores"], flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(out / "predictions.npz", actual=actual, **predictions)


if __name__ == "__main__":
    main()
