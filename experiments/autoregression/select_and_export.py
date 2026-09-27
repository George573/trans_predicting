"""Select a direct/recursive blend across two origins and export final forecasts."""

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from probe import detailed
from run import FINAL, ROUTES, fit, load, load_data

from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.predict import fixed_forecast


def write_submission(path, forecast):
    if (
        forecast.shape != (9, 61, 24)
        or not np.isfinite(forecast).all()
        or (forecast < 0).any()
    ):
        raise ValueError("invalid forecast")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["route", "date", "hour", "prediction"])
        for route in sorted((*ROUTES, 5)):
            for d in range(61):
                for h in range(24):
                    value = (
                        0.0
                        if route == 5
                        else float(forecast[ROUTES.index(route), d, h])
                    )
                    writer.writerow(
                        [route, FINAL + timedelta(days=d), h, round(value, 6)]
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/autoregression_selected")
    )
    parser.add_argument("--include-tuned", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    settings = SimpleNamespace(
        history_days=28, device="cpu", leads=12, batch_size=32, lr=0.001, patience=5
    )
    july = np.load("outputs/autoregression_july/predictions.npz")
    september = np.load("outputs/autoregression_probe/predictions.npz")
    actual = [july["actual"], september["actual"]]
    components = {
        "patch_direct": [july["patch_direct"], september["patch_123_direct"]],
        "cnn_direct": [july["cnn_direct"], september["cnn_67_direct"]],
        "cnn_recursive": [july["cnn_recursive_1"], september["cnn_67_recursive_1"]],
    }
    tuned_epoch = None
    if args.include_tuned:
        tuned_root = Path("outputs/autoregression_cnn_selected")
        tuned_epoch = json.loads((tuned_root / "results.json").read_text())["epoch"]
        if tuned_epoch == 7:
            components["cnn_tuned_recursive"] = components["cnn_recursive"]
        else:
            cutoff = date(2025, 7, 2)
            model, _ = fit(
                "cnn",
                load_data(cutoff),
                settings,
                tuned_epoch,
                seed=67,
                output=args.output / "july_tuned_cnn.pt",
            )
            pred = recursive_forecast(model, load_data(cutoff), cutoff)
            components["cnn_tuned_recursive"] = [
                pred,
                np.load(tuned_root / "predictions.npz")["cnn"],
            ]
    recipes = {"previous_direct_blend": {"patch_direct": 0.75, "cnn_direct": 0.25}}
    for cnn in ("cnn_recursive", "cnn_tuned_recursive"):
        if cnn not in components:
            continue
        for weight in (0.0, 0.25, 0.5, 0.75):
            recipes[f"{cnn}_patch_{weight}"] = {"patch_direct": weight, cnn: 1 - weight}
    scores, predictions = {}, {}
    denominator = sum(a.astype(np.float64).sum() for a in actual)
    for name, recipe in recipes.items():
        predicted = [
            sum(
                weight * components[component][i]
                for component, weight in recipe.items()
            )
            for i in (0, 1)
        ]
        error = sum(
            np.abs(a.astype(np.float64) - p).sum() for a, p in zip(actual, predicted)
        )
        scores[name] = dict(
            pooled_score=1 - error / denominator,
            july=detailed(actual[0], predicted[0]),
            september=detailed(actual[1], predicted[1]),
        )
        predictions[name] = predicted
    winner = max(scores, key=lambda k: scores[k]["pooled_score"])
    latest = max(scores, key=lambda k: scores[k]["september"]["score"])
    baseline = scores["previous_direct_blend"]
    both = [
        k
        for k, v in scores.items()
        if v["july"]["score"] >= baseline["july"]["score"]
        and v["september"]["score"] >= baseline["september"]["score"]
    ]
    conservative = max(both, key=lambda k: scores[k]["pooled_score"])
    report = dict(
        selection="maximum pooled WAPE-score across two 61-day origins",
        winner=winner,
        recipe=recipes[winner],
        latest_only=latest,
        conservative=conservative,
        tuned_epoch=tuned_epoch,
        scores=scores,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "validation_predictions.npz",
        actual_july=actual[0],
        actual_september=actual[1],
        **{
            name + "_" + str(i): p
            for name, values in predictions.items()
            for i, p in enumerate(values)
        },
    )
    print(
        json.dumps({k: v["pooled_score"] for k, v in scores.items()}, indent=2),
        flush=True,
    )
    print("selected", winner, flush=True)
    data = load_data(FINAL)
    root = Path("outputs/patch_transformer_trial")
    patch = load(root / "final_patch_123.pt")
    cnn = load(root / "final_cnn_67.pt")
    final_components = dict(
        patch_direct=fixed_forecast(patch, data, FINAL, 61, 28),
        cnn_direct=fixed_forecast(cnn, data, FINAL, 61, 14),
        cnn_recursive=recursive_forecast(cnn, data, FINAL),
    )
    sources = {
        "patch_direct": str(root / "final_patch_123.pt"),
        "cnn_direct": str(root / "final_cnn_67.pt"),
        "cnn_recursive": str(root / "final_cnn_67.pt"),
    }
    if any("cnn_tuned_recursive" in recipes[k] for k in (winner, latest, conservative)):
        if tuned_epoch == 7:
            final_components["cnn_tuned_recursive"] = final_components["cnn_recursive"]
            sources["cnn_tuned_recursive"] = sources["cnn_recursive"]
        else:
            path = args.output / "final_tuned_cnn.pt"
            tuned, _ = fit("cnn", data, settings, tuned_epoch, seed=67, output=path)
            final_components["cnn_tuned_recursive"] = recursive_forecast(
                tuned, data, FINAL
            )
            sources["cnn_tuned_recursive"] = str(path)
    exports = {}
    for filename, selection in [
        ("submission.csv", winner),
        ("submission_conservative.csv", conservative),
        ("submission_latest_holdout.csv", latest),
    ]:
        values = sum(
            weight * final_components[name]
            for name, weight in recipes[selection].items()
        )
        write_submission(args.output / filename, values)
        exports[filename] = dict(
            selection=selection, recipe=recipes[selection], rows=14640
        )
    np.savez_compressed(args.output / "final_components.npz", **final_components)
    (args.output / "submission_metadata.json").write_text(
        json.dumps(
            dict(
                exports=exports,
                checkpoints=sources,
                cutoff=str(FINAL),
                days=61,
                note="Recursive components append predictions only; no November/December labels are available.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
