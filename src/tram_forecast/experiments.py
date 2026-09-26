"""Preview or run controlled full or boarding-only architecture comparisons."""

import argparse
import json
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint import load_model
from .evaluate import evaluate_model, evaluate_weekly_profiles
from .io import write_json
from .model import ForecastNetwork
from .settings import Settings
from .storage import Store
from .train import train


def variants(suite):
    base = Settings.from_dict(suite["base"])
    seeds = suite.get("seeds", [base.model.seed])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    names = set()
    result = []
    for variant in suite["variants"]:
        name = variant["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name) or name in names:
            raise ValueError("variant names must be unique lowercase names without path separators")
        names.add(name)
        for seed in seeds:
            model = replace(base.model, **variant.get("model", {}), seed=seed)
            if model.category_caps != base.model.category_caps or model.min_frequency != base.model.min_frequency:
                raise ValueError("suite variants must share fitted vocabularies")
            first_context = base.training.context_start_days
            if first_context is None and model.history_days != base.model.history_days:
                raise ValueError("suite variants must share history length or set context_start_days")
            if first_context is not None and first_context < model.history_days:
                raise ValueError("context_start_days must cover every variant's history")
            result.append((f"{name}_seed{seed}", replace(base, model=model)))
    if not result:
        raise ValueError("suite must contain variants")
    return result


def run_suite(
    suite_path, artifact=None, output="outputs/architecture", device=None,
    run=False, overwrite=False,
):
    """Train fresh runs; overwrite archives existing selected runs before restarting."""
    suite = json.loads(Path(suite_path).read_text())
    runs = variants(suite)
    model_kind = suite.get("model_kind", "full")
    if model_kind not in ("full", "boarding_only"):
        raise ValueError("model_kind must be full or boarding_only")
    if run and artifact is None:
        raise ValueError("--artifact is required with --run")
    store = Store(artifact, events=model_kind == "full") if artifact else None
    output = Path(output)
    existing = [output / name for name, _ in runs
                if (output / name).exists() or (output / name).is_symlink()]
    for filename in ("comparison.json", "baselines.json"):
        path = output / filename
        if path.exists() or path.is_symlink():
            existing.append(path)
    if run and existing:
        if not overwrite:
            raise ValueError(
                "experiment run directory already exists; choose a new --output "
                "or pass overwrite=True (--overwrite) to archive and restart"
            )
        archive_root = output / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        archive = Path(tempfile.mkdtemp(prefix=stamp, dir=archive_root))
        for path in existing:
            path.rename(archive / path.name)
        print(f"Previous results archived to {archive}", flush=True)
    baseline_report = None
    if run:
        baseline_report = evaluate_weekly_profiles(store, runs[0][1].training.forecast_days)
        write_json(output / "baselines.json", baseline_report)
    rows = []
    for name, settings in runs:
        if device:
            settings = replace(settings, training=replace(settings.training, device=device))
        vocab = store.metadata["vocab_sizes"] if store else [c + 3 for c in settings.model.category_caps]
        model = ForecastNetwork(vocab, store.metadata["scale"] if store else 1, settings.model, model_kind)
        row = {"name": name, "model_kind": model_kind, "variant": name.rsplit("_seed", 1)[0],
               "seed": settings.model.seed, "parameters": model.parameter_report()["total"],
               "parameter_basis": "artifact vocabulary" if store else "maximum vocabulary",
               "history_days": settings.model.history_days,
               "encoded_width": model.encoded_width,
               "mlp_input_width": model.encoded_width + 13,
               "event_output_width": model.event_width,
               "event_parameters": model.parameter_report()["events"],
               "forecast_days": settings.training.forecast_days,
               "settings": settings.to_dict()}
        del model
        print(
            f"{name}: {row['parameters']:,} parameters ({row['parameter_basis']}) | "
            f"history features={row['encoded_width']} | MLP input={row['mlp_input_width']} | "
            f"event output={row['event_output_width']}", flush=True,
        )
        if run:
            directory = output / name
            checkpoint = train(settings, artifact, model_kind, directory)
            model, payload = load_model(checkpoint, store, settings.training.device)
            report = evaluate_model(model, store, settings.training.forecast_days)
            write_json(directory / "evaluation.json", report)
            history = json.loads((directory / "history.json").read_text())
            row.update(wape=report["neural"]["global"]["wape"],
                       mae=report["neural"]["global"]["mae"],
                       baseline_wape=next(
                           b["wape"] for b in baseline_report["baselines"]
                           if b["history_days"] == settings.model.history_days
                       ),
                       weekly_baselines={str(b["history_days"]): b for b in baseline_report["baselines"]},
                       best_epoch=payload["progress"]["best_epoch"],
                       epochs_completed=len(history),
                       training_seconds=sum(e["seconds"] for e in history),
                       checkpoint=str(checkpoint))
            del model
        rows.append(row)
        if run:
            write_json(output / "comparison.json", sorted(rows, key=lambda r: r["wape"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/experiments/full_week.json")
    parser.add_argument("--artifact")
    parser.add_argument("--output", default="outputs/architecture")
    parser.add_argument("--device")
    parser.add_argument("--run", action="store_true", help="perform training; otherwise only preview parameter counts")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="archive existing selected runs and train from scratch (requires --run)",
    )
    args = parser.parse_args()
    run_suite(args.suite, args.artifact, args.output, args.device, args.run, args.overwrite)


if __name__ == "__main__":
    main()
