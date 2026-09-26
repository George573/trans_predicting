"""Validated, serializable model configuration; path tuples are (width, kernel, dilation)."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)


@dataclass(frozen=True)
class Config:
    hourly: tuple = ((12, 3, 1), (12, 5, 24), (8, 3, 168))
    shared1: tuple = ((24, 3, 1), (24, 5, 2), (16, 3, 12))
    shared2: tuple = ((12, 3, 1), (12, 5, 2), (8, 3, 12))
    shared3: tuple = ((6, 3, 1), (6, 5, 2), (4, 3, 6))
    hourly_stride: int = 1
    history_days: int = 21
    shared_depth: int = 3
    temporal_pool: str = "max"
    pooled_hours: int | None = None
    head_width: int = 250
    dropout: float = 0.1
    seed: int = 67

    def __post_init__(self):
        for name in ("hourly", "shared1", "shared2", "shared3"):
            paths = tuple(tuple(p) for p in getattr(self, name))
            if len(paths) < 2 or any(
                len(p) != 3
                or any(type(v) is not int or v < 1 for v in p)
                or p[1] % 2 != 1
                for p in paths
            ):
                raise ValueError(
                    f"{name}: paths require positive widths/dilations and odd kernels"
                )
            object.__setattr__(self, name, paths)
        for name in ("history_days", "head_width", "hourly_stride"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.shared_depth) is not int or self.shared_depth not in (1, 2, 3):
            raise ValueError("shared_depth must be 1, 2 or 3")
        if self.temporal_pool not in ("max", "avg"):
            raise ValueError("pooling must be max or avg")
        length = (self.history_days * 24 + self.hourly_stride - 1) // self.hourly_stride
        for factor in (2, 2, 3)[:self.shared_depth]:
            length //= factor
        if length < 1 or (self.pooled_hours is not None and (
            type(self.pooled_hours) is not int or not 1 <= self.pooled_hours <= length
        )):
            raise ValueError("pooled_hours must fit the downsampled history")
        if not 0 <= self.dropout < 1:
            raise ValueError("invalid dropout")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
