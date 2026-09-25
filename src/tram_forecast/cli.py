"""Explicit application commands; no side effects on import."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .io import digest, write_json
from .settings import Settings


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed-cutoff tram forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("check-config", "Validate configuration"),
        ("prepare", "Build immutable disk artifacts"),
        ("inspect", "Read artifact metadata"),
        ("smoke", "Forward/backward resource check, without optimizer steps"),
        ("train", "Train a validation model"),
        ("evaluate", "Evaluate checkpoint and weekly baseline"),
        ("compare", "Compare completed validation runs"),
        ("refit", "Fit selected model from scratch on final observations"),
        ("predict", "Write validated final submission"),
    ]:
        p = commands.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.json")
        if name not in ("check-config", "compare"):
            p.add_argument("--artifact", required=name != "prepare")
        if name == "prepare":
            p.add_argument("--regime", choices=["validation", "final"], required=True)
        if name in ("train", "smoke"):
            p.add_argument("--model", choices=["boarding_only", "full"], default="full")
        if name in ("train", "refit"):
            p.add_argument("--resume")
        if name in ("evaluate", "predict"):
            p.add_argument("--checkpoint", required=True)
        if name == "refit":
            p.add_argument("--selected-checkpoint", required=True)
        if name == "predict":
            p.add_argument("--template", required=True)
        if name == "compare":
            p.add_argument("--reports", nargs="+", required=True)
        if name in ("smoke", "train", "evaluate", "compare", "refit", "predict"):
            p.add_argument("--output", required=name == "predict")
        if name in ("smoke", "train", "evaluate", "refit", "predict"):
            p.add_argument("--device")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("check-config", "prepare", "train", "smoke"):
            settings = Settings.load(args.config)
            if getattr(args, "device", None):
                settings = replace(
                    settings, training=replace(settings.training, device=args.device)
                )
        if args.command == "check-config":
            result = {
                "fingerprint": digest(settings.to_dict()),
                "resolved": settings.to_dict(),
            }
        elif args.command == "prepare":
            from .preprocess import prepare

            result = {"artifact": str(prepare(settings, args.regime, args.artifact))}
        elif args.command == "inspect":
            from .storage import Store

            result = Store(args.artifact, events=False).metadata
        elif args.command == "smoke":
            from .smoke import smoke

            result = smoke(settings, args.artifact, args.model)
        elif args.command == "train":
            from .train import train

            result = {
                "checkpoint": str(
                    train(settings, args.artifact, args.model, args.output, args.resume)
                )
            }
        elif args.command == "evaluate":
            from .checkpoint import load_model, read_checkpoint
            from .evaluate import evaluate_model
            from .storage import Store

            payload = read_checkpoint(args.checkpoint)
            store = Store(args.artifact, events=payload["model_kind"] == "full")
            model, _ = load_model(args.checkpoint, store, args.device or "cpu")
            result = evaluate_model(model, store)
        elif args.command == "compare":
            reports = [json.loads(Path(p).read_text()) for p in args.reports]
            if (
                len({r["artifact"] for r in reports}) != 1
                or len({r["cutoff"] for r in reports}) != 1
            ):
                raise ValueError(
                    "comparison requires identical evaluation artifacts/cutoffs"
                )
            result = {
                "runs": [
                    {
                        "path": p,
                        "model_kind": r["model_kind"],
                        "metrics": r["neural"]["global"],
                    }
                    for p, r in zip(args.reports, reports)
                ],
                "weekly_profile": reports[0]["weekly_profile"]["global"],
            }
        elif args.command == "refit":
            from .train import refit

            result = {
                "checkpoint": str(
                    refit(
                        args.selected_checkpoint,
                        args.artifact,
                        args.output,
                        args.device,
                        args.resume,
                    )
                )
            }
        else:
            from .predict import predict

            result = {
                "submission": str(
                    predict(
                        args.checkpoint,
                        args.artifact,
                        args.template,
                        args.output,
                        args.device or "cpu",
                    )
                )
            }
        if getattr(args, "output", None) and args.command in (
            "smoke",
            "evaluate",
            "compare",
        ):
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
