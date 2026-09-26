"""Explicit training commands. Importing this module never trains a model."""

import math
import time
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .checkpoint import (
    load_model,
    read_checkpoint,
    restore_rng,
    save_checkpoint,
    seed_all,
)
from .data import ForecastDataset, collate_samples, epoch_order
from .evaluate import evaluate_model
from .io import write_json
from .model import ForecastNetwork
from .config import Settings


def mae(prediction, target):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 24
    ):
        raise ValueError("loss expects matching [requests,24] predictions/targets")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite regression values")
    return (prediction - target).abs().mean()


def normalize_gradients(parameters, sample_count):
    if sample_count < 1:
        raise ValueError("empty accumulation group")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)


class TrainingProgress:
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.bar = None

    @contextmanager
    def epoch(self, epoch, epochs, total):
        with tqdm(
            total=total,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            file=self.stream,
            dynamic_ncols=True,
            mininterval=0.5,
        ) as bar:
            self.bar = bar
            try:
                yield
            finally:
                self.bar = None

    def show(self, message, force=False):
        self.message(message)

    def message(self, message):
        tqdm.write(message, file=self.stream)

    def batch(self, completed, mae, steps):
        self.bar.set_postfix(MAE=f"{mae:.5g}", step=steps, refresh=False)
        self.bar.update(completed - self.bar.n)


def train(settings, artifact, output=None, resume=None, final=False):
    seed_all(settings.training.seed)
    dataset = ForecastDataset(
        artifact, settings.training.forecast_days,
    )
    store = dataset.store
    if store.metadata["regime"] != ("final" if final else "validation"):
        raise ValueError("training regime mismatch")
    if not len(dataset):
        raise ValueError("no eligible training identities")
    if not final:
        from datetime import date

        available = (date.fromisoformat(store.metadata["evaluation_end"]) - store.end).days
        if settings.training.forecast_days > available:
            raise ValueError("forecast_days exceeds available validation days")
    expected_end = (
        settings.data.final_cutoff if final else settings.data.validation_cutoff
    )
    if (
        str(store.start) != settings.data.start
        or str(store.end) != expected_end
        or store.routes != settings.data.routes
    ):
        raise ValueError("training artifact dates/routes disagree with configuration")
    output = Path(
        output
        or (Path(resume).resolve().parent if resume else None)
        or Path(settings.training.output_root)
        / ("final" if final else "validation")
        / "boarding_only"
    )
    output.mkdir(parents=True, exist_ok=True)
    if resume and output.resolve() != Path(resume).resolve().parent:
        raise ValueError("epoch-boundary resume must use the original run directory")
    if not resume and (
        (output / "latest.pt").exists() or (output / "best.pt").exists()
    ):
        raise ValueError("run already exists; use --resume or a new output directory")
    model = ForecastNetwork(store.metadata["scale"]).to(settings.training.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings.training.weight_decay,
    )
    epoch_start = 0
    step = 0
    best = None if final else math.inf
    best_epoch = 0
    stale = 0
    if resume:
        loaded, payload = load_model(resume, store, settings.training.device)
        old = Settings.from_dict(payload["settings"])
        old_training = old.to_dict()["training"]
        new_training = settings.to_dict()["training"]
        for key in ("epochs", "output_root"):
            old_training.pop(key)
            new_training.pop(key)
        if old_training != new_training:
            raise ValueError("resume configuration mismatch")
        model.load_state_dict(loaded.state_dict())
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer for resume")
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        progress = payload["progress"]
        epoch_start = progress["epoch"]
        step = progress["step"]
        best = progress["best"]
        best_epoch = progress["best_epoch"]
        stale = progress["stale"]
        if not final and stale >= settings.training.patience:
            if not (output / "best.pt").exists():
                raise ValueError("resume run is missing its best checkpoint")
            return output / "best.pt"
    write_json(output / "config.json", settings.to_dict())
    write_json(output / "parameters.json", model.parameter_report())
    history = []
    history_path = output / "history.json"
    if resume and history_path.exists():
        import json

        history = json.loads(history_path.read_text())
        history = [entry for entry in history if entry["epoch"] <= epoch_start]
    feedback = TrainingProgress()
    feedback.message(
        f"{'Resuming' if resume else 'Training'} boarding_only on {settings.training.device} | "
        f"{len(dataset)} contexts | batch size {settings.training.batch_size} | "
        f"accumulation {settings.training.accumulation} | output: {output}"
    )
    feedback.message(
        "Interrupt with Ctrl+C / Stop kernel. Resume from latest.pt to rerun the "
        "partial epoch; a checkpoint is available after the first completed epoch."
    )
    for epoch in range(epoch_start, settings.training.epochs):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = epoch_order(len(dataset), epoch, settings.training.seed)
        accumulated = 0
        microbatches = 0
        loss_sum = 0.0
        seen = 0
        batch_size = settings.training.batch_size
        total_batches = math.ceil(len(order) / batch_size)
        with feedback.epoch(epoch + 1, settings.training.epochs, total_batches):
            for start in range(0, len(order), batch_size):
                samples = [dataset[int(i)] for i in order[start : start + batch_size]]
                inputs, request, target = collate_samples(samples, settings.training.device)
                loss = mae(model(inputs, request), target)
                # Weight each requested day equally, including partial end-of-period horizons.
                count = target.shape[0]
                (loss * count).backward()
                accumulated += count
                microbatches += 1
                seen += count
                loss_sum += float(loss.detach()) * count
                if (
                    microbatches == settings.training.accumulation
                    or start + batch_size >= len(order)
                ):
                    normalize_gradients(model.parameters(), accumulated)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        settings.training.clip_norm,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    accumulated = 0
                    microbatches = 0
                feedback.batch(
                    start // batch_size + 1, loss_sum / seen, step,
                )
        report = None
        improved = False
        if not final:
            feedback.show(f"Epoch {epoch + 1}: validating…", force=True)
            report = evaluate_model(model, store, settings.training.forecast_days)
            score = report["neural"]["global"]["wape"]
            if score is None:
                raise ValueError(
                    "validation WAPE undefined; cannot select a checkpoint"
                )
            improved = score < best
            if improved:
                best = score
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch + 1
        progress = {
            "epoch": epoch + 1,
            "step": step,
            "best": best,
            "best_epoch": best_epoch,
            "stale": stale,
        }
        feedback.show(f"Epoch {epoch + 1}: saving checkpoints…", force=True)
        save_checkpoint(
            output / "latest.pt", model, store, settings, optimizer, **progress
        )
        if improved:
            save_checkpoint(
                output / "best.pt", model, store, settings, optimizer, **progress
            )
        if final:
            save_checkpoint(
                output / "final.pt", model, store, settings, optimizer, **progress
            )
        history.append(
            {
                "epoch": epoch + 1,
                "mae": loss_sum / seen,
                "seconds": time.monotonic() - started,
                "validation": report,
            }
        )
        write_json(history_path, history)
        validation = (
            f"WAPE={score:.4%} | best={best:.4%} (epoch {best_epoch}) | "
            f"patience={stale}/{settings.training.patience}"
            if not final else "final refit"
        )
        feedback.message(
            f"Epoch {epoch + 1}/{settings.training.epochs} complete | "
            f"MAE={loss_sum / seen:.6g} | {validation} | "
            f"{history[-1]['seconds']:.1f}s | "
            f"{'new best saved; ' if improved else ''}latest.pt saved"
        )
        if not final and stale >= settings.training.patience:
            feedback.message(f"Early stopping: no improvement for {stale} epochs.")
            break
    result = output / ("final.pt" if final else "best.pt")
    if not result.exists():
        raise ValueError(
            "no completed epochs/checkpoint; check resume epoch and output directory"
        )
    feedback.message(f"Training complete. Checkpoint: {result}")
    return result


def refit(selected_checkpoint, artifact, output=None, device=None, resume=None, epochs=None):
    payload = read_checkpoint(selected_checkpoint)
    if (
        payload["artifact_contract"]["end"]
        != payload["settings"]["data"]["validation_cutoff"]
    ):
        raise ValueError("refit requires a validation-selected checkpoint")
    selected_epoch = payload["progress"].get("best_epoch", 0)
    epochs = selected_epoch if epochs is None else epochs
    if type(epochs) is not int or epochs < 1:
        raise ValueError("refit epochs must be a positive integer")
    settings = Settings.from_dict(payload["settings"])
    training = replace(
        settings.training, epochs=epochs, device=device or settings.training.device
    )
    settings = replace(settings, training=training)
    from .data import Store

    store = Store(artifact)
    if (
        str(store.end) != settings.data.final_cutoff
        or str(store.start) != settings.data.start
        or store.routes != settings.data.routes
    ):
        raise ValueError("final refit artifact period/routes mismatch")
    return train(
        settings, artifact, output, resume=resume, final=True
    )
