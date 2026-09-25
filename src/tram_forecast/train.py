"""Explicit training commands. Importing this module never trains a model."""

import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from .checkpoint import (
    load_model,
    read_checkpoint,
    restore_rng,
    save_checkpoint,
    seed_all,
)
from .dataset import ForecastDataset, collate_samples, epoch_order
from .evaluate import evaluate_model
from .io import write_json
from .losses import mae, normalize_gradients
from .model import ForecastNetwork
from .settings import Settings


def train(settings, artifact, model_kind="full", output=None, resume=None, final=False):
    seed_all(settings.model.seed)
    dataset = ForecastDataset(artifact, model_kind, settings.training.forecast_days)
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
    preparation = store.metadata["identity"]
    expected_end = (
        settings.data.final_cutoff if final else settings.data.validation_cutoff
    )
    if (
        str(store.start) != settings.data.start
        or str(store.end) != expected_end
        or store.routes != settings.data.routes
    ):
        raise ValueError("training artifact dates/routes disagree with configuration")
    if preparation["min_frequency"] != settings.model.min_frequency or list(
        preparation["category_caps"]
    ) != list(settings.model.category_caps):
        raise ValueError("configuration disagrees with fitted vocabularies")
    output = Path(
        output
        or (Path(resume).resolve().parent if resume else None)
        or Path(settings.training.output_root)
        / ("final" if final else "validation")
        / model_kind
    )
    output.mkdir(parents=True, exist_ok=True)
    if resume and output.resolve() != Path(resume).resolve().parent:
        raise ValueError("epoch-boundary resume must use the original run directory")
    if not resume and (
        (output / "latest.pt").exists() or (output / "best.pt").exists()
    ):
        raise ValueError("run already exists; use --resume or a new output directory")
    model = ForecastNetwork(
        store.metadata["vocab_sizes"],
        store.metadata["scale"],
        settings.model,
        model_kind,
    ).to(settings.training.device)
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
        if payload.get("training_layout") != "grouped_contexts_v1":
            raise ValueError("legacy training layout cannot resume; start a fresh run")
        old = Settings.from_dict(payload["settings"])
        old_training = old.to_dict()["training"]
        new_training = settings.to_dict()["training"]
        for key in ("epochs", "output_root"):
            old_training.pop(key)
            new_training.pop(key)
        if (
            old.model != settings.model
            or old_training != new_training
            or payload["model_kind"] != model_kind
        ):
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
    for epoch in range(epoch_start, settings.training.epochs):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = epoch_order(len(dataset), epoch, settings.model.seed)
        accumulated = 0
        microbatches = 0
        loss_sum = 0.0
        seen = 0
        batch_size = settings.training.batch_size
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
        report = None
        improved = False
        if not final:
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
        print(
            f"Epoch {epoch + 1}: MAE={loss_sum / seen:.6g}, WAPE={best if not final else 'not evaluated'}",
            flush=True,
        )
        if not final and stale >= settings.training.patience:
            break
    result = output / ("final.pt" if final else "best.pt")
    if not result.exists():
        raise ValueError(
            "no completed epochs/checkpoint; check resume epoch and output directory"
        )
    return result


def refit(selected_checkpoint, artifact, output=None, device=None, resume=None):
    payload = read_checkpoint(selected_checkpoint)
    if (
        payload["artifact_contract"]["end"]
        != payload["settings"]["data"]["validation_cutoff"]
    ):
        raise ValueError("refit requires a validation-selected checkpoint")
    epochs = payload["progress"].get("best_epoch", 0)
    if epochs < 1:
        raise ValueError("checkpoint has no selected validation epoch")
    settings = Settings.from_dict(payload["settings"])
    training = replace(
        settings.training, epochs=epochs, device=device or settings.training.device
    )
    settings = replace(settings, training=training)
    from .storage import Store

    store = Store(artifact, events=False)
    if (
        str(store.end) != settings.data.final_cutoff
        or str(store.start) != settings.data.start
        or store.routes != settings.data.routes
    ):
        raise ValueError("final refit artifact period/routes mismatch")
    return train(
        settings, artifact, payload["model_kind"], output, resume=resume, final=True
    )
