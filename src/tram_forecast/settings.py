"""Application settings, separate from checkpoint-compatible model configuration."""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .config import ROUTES, Config


@dataclass(frozen=True)
class DataSettings:
    raw_paths: tuple = ("dataset/train.csv", "dataset/test.csv")
    label_paths: tuple = (
        "dataset/labels/labels_day_train.csv",
        "dataset/labels/labels_day_test.csv",
    )
    routes: tuple = ROUTES
    start: str = "2025-01-01"
    validation_cutoff: str = "2025-09-01"
    final_cutoff: str = "2025-11-01"
    forecast_end: str = "2026-01-01"
    output_root: str = "outputs/prepared"
    temp_dir: str = "outputs/tmp"
    memory_limit: str = "2GB"
    threads: int = 2
    fetch_rows: int = 10000
    min_free_disk_bytes: int = 1_000_000_000

    def __post_init__(self):
        for name in ("raw_paths", "label_paths", "routes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.raw_paths
            or not self.label_paths
            or not self.routes
            or len(set(self.routes)) != len(self.routes)
            or any(r not in ROUTES for r in self.routes)
        ):
            raise ValueError(
                "nonempty paths and unique supported neural routes required"
            )
        dates = [
            date.fromisoformat(getattr(self, n))
            for n in ("start", "validation_cutoff", "final_cutoff", "forecast_end")
        ]
        if (
            not all(a < b for a, b in zip(dates, dates[1:]))
            or (dates[1] - dates[0]).days < 22
            or (dates[2] - dates[1]).days > 61
            or (dates[3] - dates[2]).days > 61
        ):
            raise ValueError(
                "ordered fitting/evaluation dates with at least 22 training days required"
            )
        if self.threads < 1 or self.fetch_rows < 1 or self.min_free_disk_bytes < 0:
            raise ValueError("invalid preparation resource limits")


@dataclass(frozen=True)
class TrainSettings:
    forecast_days: int = 7
    batch_size: int = 1
    accumulation: int = 8
    epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    clip_norm: float = 1.0
    device: str = "cpu"
    output_root: str = "outputs/runs"
    num_workers: int = 0
    context_start_days: int | None = None

    def __post_init__(self):
        if type(self.forecast_days) is not int or not 1 <= self.forecast_days <= 61:
            raise ValueError("forecast_days must be an integer in 1..61")
        if any(
            type(v) is not int or v < 1
            for v in (self.batch_size, self.accumulation, self.epochs, self.patience)
        ):
            raise ValueError("training counts must be positive integers")
        if self.context_start_days is not None and (
            type(self.context_start_days) is not int or self.context_start_days < 1
        ):
            raise ValueError("context_start_days must be a positive integer or null")
        if self.num_workers != 0:
            raise ValueError("v1 requires num_workers=0 to bound prefetch memory")
        if (
            any(
                not math.isfinite(v) or v <= 0
                for v in (self.learning_rate, self.clip_norm)
            )
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("invalid optimizer settings")


@dataclass(frozen=True)
class Settings:
    model: Config = field(default_factory=Config)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainSettings = field(default_factory=TrainSettings)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {"model", "data", "training"}
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(
            Config(**value.get("model", {})),
            DataSettings(**value.get("data", {})),
            TrainSettings(**value.get("training", {})),
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
