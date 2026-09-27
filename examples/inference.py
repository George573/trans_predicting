"""Run from the repository root after installing the Python package."""

import argparse
from datetime import date, timedelta

from tram_forecast.data import Boardings
from tram_forecast.runner import InferenceRunner
from tram_forecast.train import load_model


def main():
    parser = argparse.ArgumentParser(description="Example context/day inference API")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--route", type=int, default=7)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    boardings = Boardings.load(
        args.labels, routes=(args.route,), events_path=args.events,
        start=args.cutoff - timedelta(days=14), end=args.cutoff,
    )
    runner = InferenceRunner(load_model(args.checkpoint))

    # Reuse this context for any target day within its 61-day horizon.
    context = runner.apply_context(boardings, args.route, args.cutoff)
    print("Reused context:", runner.predict_day(context, args.day).tolist())

    # The convenience method prepares a fresh context for this single call.
    print("Combined call:", runner.predict(boardings, args.route, args.cutoff, args.day).tolist())


if __name__ == "__main__":
    main()
