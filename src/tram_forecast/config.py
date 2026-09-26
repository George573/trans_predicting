"""Validated, serializable model configuration; path tuples are (width, kernel, dilation)."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDS = ("validation_result", "tran_type_id", "good_type", "place_id", "pass_route")
ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)


@dataclass(frozen=True)
class Config:
    embedding_dims: tuple = (4, 4, 8, 4, 8)
    category_caps: tuple = (32, 128, 512, 32, 8192)
    min_frequency: int = 5
    event1: tuple = ((12, 3, 1), (12, 5, 1), (8, 3, 2))
    event2: tuple = ((12, 3, 1), (12, 5, 2), (8, 3, 4))
    hourly: tuple = ((12, 3, 1), (12, 5, 24), (8, 3, 168))
    shared1: tuple = ((24, 3, 1), (24, 5, 2), (16, 3, 12))
    shared2: tuple = ((12, 3, 1), (12, 5, 2), (8, 3, 12))
    shared3: tuple = ((6, 3, 1), (6, 5, 2), (4, 3, 6))
    hourly_stride: int = 1
    history_days: int = 21
    event_depth: int = 2
    shared_depth: int = 3
    event_pool: str = "max"
    temporal_pool: str = "max"
    pooled_hours: int | None = None
    head_width: int = 250
    max_hours: int = 32
    max_positions: int = 32768
    checkpoint_events: bool = True
    dropout: float = 0.1
    seed: int = 67

    def __post_init__(self):
        for name in ("embedding_dims", "category_caps"):
            values = tuple(getattr(self, name))
            if len(values) != 5 or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f"{name} must contain five positive integers")
            object.__setattr__(self, name, values)
        for name, _ in [
            ("event1", 32),
            ("event2", 32),
            ("hourly", 32),
            ("shared1", 64),
            ("shared2", 32),
            ("shared3", 16),
        ]:
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
        if type(self.event_depth) is not int or self.event_depth not in (1, 2):
            raise ValueError("event_depth must be 1 or 2")
        if type(self.shared_depth) is not int or self.shared_depth not in (1, 2, 3):
            raise ValueError("shared_depth must be 1, 2 or 3")
        if self.event_pool not in ("max", "avg") or self.temporal_pool not in ("max", "avg"):
            raise ValueError("pooling must be max or avg")
        length = (self.history_days * 24 + self.hourly_stride - 1) // self.hourly_stride
        for factor in (2, 2, 3)[:self.shared_depth]:
            length //= factor
        if length < 1 or (self.pooled_hours is not None and (
            type(self.pooled_hours) is not int or not 1 <= self.pooled_hours <= length
        )):
            raise ValueError("pooled_hours must fit the downsampled history")
        if (
            type(self.max_hours) is not int
            or self.max_hours < 1
            or type(self.max_positions) is not int
            or self.max_positions <= 2 * self.halo
        ):
            raise ValueError("chunk budget must allow a positive core plus both halos")
        if not 0 <= self.dropout < 1 or self.min_frequency < 1:
            raise ValueError("invalid dropout or minimum category frequency")

    @property
    def halo(self):
        return sum(
            max((k - 1) * d // 2 for _, k, d in paths)
            for paths in (self.event1, self.event2)[:self.event_depth]
        )

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
