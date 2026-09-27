"""Train short-step and free-running multistep models; select by 61-day WAPE."""

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patch_transformer"))
from probe import detailed
from run import START, Windows, load, load_data, save, seed_all

from tram_forecast.autoregressive import recursive_forecast
from tram_forecast.patch_model import PatchForecastNetwork


class RolloutWindows(Dataset):
    def __init__(self, data, block, steps, history_days=28):
        self.cached = Windows(data, history_days)
        self.horizon, self.history_days = block * steps, history_days
        self.index = [
            (r, c) for r, c in self.cached.index if c + self.horizon <= self.cached.days
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        r, c = self.index[index]
        h, n = self.history_days, self.horizon
        return dict(
            counts=self.cached.data.counts[r : r + 1, (c - h) * 24 : c * 24],
            calendar=self.cached.features[r, :, (c - h) * 24 : (c + n) * 24],
            request=self.cached.requests[r, c : c + n],
            route=r + 1,
            target=self.cached.data.counts[r, c * 24 : (c + n) * 24].reshape(n, 24),
        )


def rollout_loss(model, batch, block, steps, scale):
    state = batch["counts"]
    size = state.shape[0]
    horizon_hours = state.shape[-1]
    losses = []
    for step in range(steps):
        offset = step * block
        history = dict(
            counts=state[:, :, -horizon_hours:],
            calendar=batch["calendar"][:, :, offset * 24 : offset * 24 + horizon_hours],
        )
        request = dict(
            context_indices=torch.arange(size, device=state.device).repeat_interleave(
                block
            ),
            route_indices=batch["route"].repeat_interleave(block),
            calendar=batch["request"][:, offset : offset + block].flatten(0, 1),
            lead=torch.arange(
                1, block + 1, device=state.device, dtype=torch.float32
            ).repeat(size),
        )
        predicted = model(history, request).reshape(size, block, 24)
        losses.append(
            (predicted - batch["target"][:, offset : offset + block]).abs().mean()
            / scale
        )
        # Keep gradients through generated history: subsequent errors train earlier steps.
        state = torch.cat((state, predicted.reshape(size, 1, -1)), dim=-1)
    return torch.stack(losses).mean()


def fit(args, data, validation, block, steps, seed, epochs, output):
    seed_all(seed)
    model = PatchForecastNetwork(data.scale).to(args.device)
    loader = DataLoader(RolloutWindows(data, block, steps), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best, stale, best_epoch = float("inf"), 0, 0
    trace = []
    cutoff = date.fromisoformat(args.cutoff)
    actual = (
        None
        if validation is None
        else validation.counts[:, (cutoff - START).days * 24 :].reshape(9, 61, 24)
    )
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = rollout_loss(model, batch, block, steps, data.scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch["route"]) * data.scale
            count += len(batch["route"])
        row = dict(
            epoch=epoch, train_mae=total / count, seconds=time.monotonic() - start
        )
        if validation is not None:
            pred = recursive_forecast(model, validation, cutoff, block_days=block)
            row.update(detailed(actual, pred))
            if row["wape"] < best:
                best, stale, best_epoch = row["wape"], 0, epoch
                save(model, "patch", 28, epoch, seed, output)
            else:
                stale += 1
        else:
            best_epoch = epoch
        trace.append(row)
        print(json.dumps(dict(block=block, steps=steps, seed=seed, **row)), flush=True)
        if validation is not None and stale >= args.patience:
            break
    if validation is None:
        save(model, "patch", 28, epochs, seed, output)
    Path(str(output) + ".history.json").write_text(json.dumps(trace, indent=2) + "\n")
    return load(output, args.device), best_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=Path("outputs/autoregression_trained")
    )
    p.add_argument("--cutoff", default="2025-09-01")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["daily", "weekly", "rollout"],
        default=["daily", "weekly", "rollout"],
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    torch.set_num_threads(args.threads)
    cutoff = date.fromisoformat(args.cutoff)
    training, validation = load_data(cutoff), load_data(cutoff + timedelta(days=61))
    actual = validation.counts[:, (cutoff - START).days * 24 :].reshape(9, 61, 24)
    results, predictions = {}, {}
    for name in args.modes:
        block, steps = {"daily": (1, 1), "weekly": (7, 1), "rollout": (7, 4)}[name]
        model, epoch = fit(
            args,
            training,
            validation,
            block,
            steps,
            args.seed,
            args.epochs,
            args.output / f"{name}.pt",
        )
        pred = recursive_forecast(model, validation, cutoff, block_days=block)
        predictions[name] = pred
        results[name] = dict(
            **detailed(actual, pred), epoch=epoch, block=block, training_steps=steps
        )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(
            args.output / "predictions.npz", actual=actual, **predictions
        )


if __name__ == "__main__":
    main()
