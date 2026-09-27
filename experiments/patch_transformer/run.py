"""Reproducible CPU/GPU comparison, fixed-cutoff selection, refit and submission.

Run from the repository root: python experiments/patch_transformer/run.py --help
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tram_forecast.data import ROUTES, Boardings, calendar, collate_samples
from tram_forecast.model import ForecastNetwork
from tram_forecast.patch_model import PatchForecastNetwork
from tram_forecast.predict import fixed_forecast

START = date(2025, 1, 1)
CUTOFF = date(2025, 9, 1)
FINAL = date(2025, 11, 1)
EVENTS = "dataset/parking_events_hourly.csv"


def load_data(end):
    # Each file is clipped before constructing training windows/scales.
    first = Boardings.load(
        "dataset/labels/labels_day_train.csv", start=START, end=end, events_path=EVENTS
    )
    second = Boardings.load(
        "dataset/labels/labels_day_test.csv", start=START, end=end, events_path=EVENTS
    )
    first.counts += second.counts
    first.scale = max(1.0, float(first.counts.mean()))
    return first


class Windows(Dataset):
    """Cached features; all observed cutoffs, including partial target horizons."""

    def __init__(self, data, history_days, leads=12):
        self.data, self.history_days, self.leads = data, history_days, leads
        self.days = data.counts.shape[1] // 24
        times = [
            datetime.combine(START, datetime.min.time()) + timedelta(hours=i)
            for i in range(self.days * 24)
        ]
        cal = calendar(times)
        self.features = np.stack(
            [
                np.concatenate(
                    (cal, data.scheduled_events.window(r, START, self.days * 24))
                )
                for r in data.routes
            ]
        )
        self.requests = np.concatenate(
            (
                np.broadcast_to(cal[2:, ::24].T, (len(data.routes), self.days, 7)),
                self.features[:, -3:]
                .reshape(len(data.routes), 3, self.days, 24)
                .max(-1)
                .transpose(0, 2, 1),
            ),
            -1,
        ).astype(np.float32)
        self.index = [
            (r, c)
            for r in range(len(data.routes))
            for c in range(history_days, self.days)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        r, c = self.index[index]
        n = min(61, self.days - c)
        offsets = np.sort(np.random.choice(n, min(n, self.leads), replace=False))
        return dict(
            counts=self.data.counts[r : r + 1, (c - self.history_days) * 24 : c * 24],
            calendar=self.features[r, :, (c - self.history_days) * 24 : c * 24],
            route_index=r + 1,
            request_calendar=self.requests[r, c + offsets],
            lead=(offsets + 1).astype(np.float32),
            target=self.data.counts[r].reshape(-1, 24)[c + offsets],
        )


def metric(actual, predicted):
    error = np.abs(actual.astype(np.float64) - predicted)
    wape = float(error.sum() / actual.sum())
    return dict(
        wape=wape,
        score=max(0.0, 1 - wape),
        mae=float(error.mean()),
        per_route_wape={
            str(r): float(error[i].sum() / actual[i].sum())
            for i, r in enumerate(ROUTES)
        },
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(
    kind,
    data,
    args,
    epochs,
    validation=None,
    seed=67,
    output=None,
    validation_forecast=None,
    validation_cutoff=CUTOFF,
):
    seed_all(seed)
    model = (
        PatchForecastNetwork(data.scale, history_days=args.history_days)
        if kind == "patch"
        else ForecastNetwork(data.scale)
    ).to(args.device)
    history_days = args.history_days if kind == "patch" else 14
    loader = DataLoader(
        Windows(data, history_days, args.leads),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, best_epoch, stale, trace = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        before = time.monotonic()
        model.train()
        absolute, positions = 0.0, 0
        for history, request, target in loader:
            history = {k: v.to(args.device) for k, v in history.items()}
            request = {k: v.to(args.device) for k, v in request.items()}
            target = target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            # Constant scaling leaves MAE/WAPE's minimizer unchanged.
            loss = (model(history, request) - target).abs().mean() / data.scale
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            absolute += loss.item() * data.scale * target.numel()
            positions += target.numel()
        row = dict(
            epoch=epoch,
            train_mae=absolute / positions,
            seconds=time.monotonic() - before,
        )
        if validation is not None:
            forecast = validation_forecast or fixed_forecast
            predicted = forecast(model, validation, validation_cutoff, 61, history_days)
            actual = validation.counts[
                :, (validation_cutoff - START).days * 24 :
            ].reshape(len(ROUTES), 61, 24)
            row.update(metric(actual, predicted))
            if row["wape"] < best:
                best, best_epoch, stale = row["wape"], epoch, 0
                save(model, kind, history_days, epoch, seed, output)
            else:
                stale += 1
        else:
            best_epoch = epoch
        trace.append(row)
        print(json.dumps(dict(model=kind, seed=seed, **row)), flush=True)
        if validation is not None and stale >= args.patience:
            break
    if validation is None:
        save(model, kind, history_days, best_epoch, seed, output)
    Path(str(output) + ".history.json").write_text(json.dumps(trace, indent=2) + "\n")
    return load(output, args.device), best_epoch


def save(model, kind, history_days, epoch, seed, path):
    torch.save(
        dict(
            architecture=kind,
            model=model.state_dict(),
            config=model.config if kind == "patch" else dict(scale=float(model.scale)),
            history_days=history_days,
            epoch=epoch,
            seed=seed,
            forecast_days=61,
        ),
        path,
    )


def load(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    cls = (
        PatchForecastNetwork if payload["architecture"] == "patch" else ForecastNetwork
    )
    model = cls(**payload["config"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/patch_transformer")
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--leads", type=int, default=12)
    parser.add_argument("--history-days", type=int, default=28)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[67, 123])
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reevaluate saved validation checkpoints; optionally refit the winner",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.leads < 1:
        parser.error("epochs and leads must be positive")
    if args.reuse_checkpoints:
        saved = json.loads((args.output / "config.json").read_text())
        for key in (
            "epochs",
            "patience",
            "batch_size",
            "leads",
            "history_days",
            "lr",
            "seeds",
            "skip_cnn",
        ):
            if saved[key] != getattr(args, key):
                parser.error(
                    f"--reuse-checkpoints requires original {key}={saved[key]}"
                )
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n"
        )
    torch.set_num_threads(args.threads)
    training, validation = load_data(CUTOFF), load_data(FINAL)
    actual = validation.counts[:, (CUTOFF - START).days * 24 :].reshape(
        len(ROUTES), 61, 24
    )
    # Include the competition's rare route 5 in headline scores (zero fallback).
    route5 = Boardings.load(
        "dataset/labels/labels_day_test.csv", routes=(5,), start=CUTOFF, end=FINAL
    )
    scores, predictions, recipes = {}, {}, {}
    for weeks in (2, 4, 8):
        weekly = training.counts[:, -weeks * 168 :].reshape(len(ROUTES), weeks, 7, 24)
        values = np.median(weekly, axis=1)[:, np.arange(61) % 7]
        predictions[f"seasonal_{weeks}w"] = values
        scores[f"seasonal_{weeks}w"] = metric(actual, values)
        recipes[f"seasonal_{weeks}w"] = {f"seasonal_{weeks}w": 1.0}
    for kind in ["cnn", "patch"] if not args.skip_cnn else ["patch"]:
        for seed in args.seeds[:1] if kind == "cnn" else args.seeds:
            name = f"{kind}_{seed}"
            checkpoint = args.output / f"{name}.pt"
            if args.reuse_checkpoints:
                model = load(checkpoint, args.device)
                epoch = torch.load(checkpoint, weights_only=True, map_location="cpu")[
                    "epoch"
                ]
            else:
                model, epoch = fit(
                    kind, training, args, args.epochs, validation, seed, checkpoint
                )
            hd = args.history_days if kind == "patch" else 14
            predictions[name] = fixed_forecast(model, validation, CUTOFF, 61, hd)
            scores[name] = dict(**metric(actual, predictions[name]), best_epoch=epoch)
            recipes[name] = {name: 1.0}
    predictions["patch_ensemble"] = np.mean(
        [predictions[f"patch_{s}"] for s in args.seeds], axis=0
    )
    scores["patch_ensemble"] = metric(actual, predictions["patch_ensemble"])
    recipes["patch_ensemble"] = {f"patch_{s}": 1 / len(args.seeds) for s in args.seeds}
    # Small, declared blend grid; validation scores are model-selection estimates.
    base_name = min(
        (k for k in scores if k.startswith("seasonal")), key=lambda k: scores[k]["wape"]
    )
    for weight in (0.25, 0.5, 0.75):
        name = f"patch_blend_{weight}"
        predictions[name] = (
            weight * predictions["patch_ensemble"]
            + (1 - weight) * predictions[base_name]
        )
        scores[name] = metric(actual, predictions[name])
        recipes[name] = {
            **{k: weight * v for k, v in recipes["patch_ensemble"].items()},
            base_name: 1 - weight,
        }
    if not args.skip_cnn:
        best_patch = min(
            [f"patch_{s}" for s in args.seeds] + ["patch_ensemble"],
            key=lambda k: scores[k]["wape"],
        )
        cnn_name = f"cnn_{args.seeds[0]}"
        for weight in (0.25, 0.5, 0.75):
            name = f"patch_cnn_blend_{weight}"
            predictions[name] = (
                weight * predictions[best_patch] + (1 - weight) * predictions[cnn_name]
            )
            scores[name] = metric(actual, predictions[name])
            recipes[name] = {
                **{k: weight * v for k, v in recipes[best_patch].items()},
                cnn_name: 1 - weight,
            }
    for name, value in predictions.items():
        total_error = (
            np.abs(actual.astype(np.float64) - value).sum() + route5.counts.sum()
        )
        wape = float(total_error / (actual.sum() + route5.counts.sum()))
        scores[name]["full_grid_wape"] = wape
        scores[name]["full_grid_score"] = max(0.0, 1 - wape)
    winner = min(scores, key=lambda k: scores[k]["wape"])
    report = dict(
        validation_cutoff=str(CUTOFF),
        validation_end="2025-10-31",
        horizon=61,
        winner=winner,
        recipe=recipes[winner],
        scores=scores,
        torch_version=torch.__version__,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "validation_predictions.npz", actual=actual, **predictions
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.refit:
        # Freeze architecture, weights and epoch counts before the final refit.
        forecasts, refit_epochs = [], {}
        for name, weight in recipes[winner].items():
            if name.startswith("seasonal_"):
                weeks = int(name.split("_")[1][:-1])
                values = np.median(
                    validation.counts[:, -weeks * 168 :].reshape(
                        len(ROUTES), weeks, 7, 24
                    ),
                    axis=1,
                )[:, np.arange(61) % 7]
            else:
                kind, seed_text = name.split("_")
                seed = int(seed_text)
                epoch = scores[name]["best_epoch"]
                model, _ = fit(
                    kind,
                    validation,
                    args,
                    epoch,
                    seed=seed,
                    output=args.output / f"final_{name}.pt",
                )
                hd = args.history_days if kind == "patch" else 14
                values = fixed_forecast(model, validation, FINAL, 61, hd)
                refit_epochs[name] = epoch
            forecasts.append(weight * values)
        forecast = np.sum(forecasts, axis=0)
        with (args.output / "submission.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["route", "date", "hour", "prediction"])
            for route in sorted((*ROUTES, 5)):
                for day in range(61):
                    for hour in range(24):
                        value = (
                            0.0
                            if route == 5
                            else float(forecast[ROUTES.index(route), day, hour])
                        )
                        if not np.isfinite(value) or value < 0:
                            raise ValueError("invalid prediction")
                        writer.writerow(
                            [route, FINAL + timedelta(days=day), hour, round(value, 6)]
                        )
        (args.output / "submission_metadata.json").write_text(
            json.dumps(
                dict(
                    selection=winner,
                    recipe=recipes[winner],
                    refit_epochs=refit_epochs,
                    rows=14640,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
