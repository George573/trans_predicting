"""Read-only artifact access with explicit fitting/evaluation separation."""

import json
from datetime import date
from pathlib import Path

import numpy as np

from .config import FIELDS
from .io import digest


class Store:
    def __init__(self, path, events=True):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        m = self.metadata
        if m["schema_version"] != 1 or digest(m["identity"]) != m["fingerprint"]:
            raise ValueError("incompatible or corrupt artifact metadata")
        self.start = date.fromisoformat(m["start"])
        self.end = date.fromisoformat(m["end"])
        self.routes = tuple(m["routes"])
        self.route_index = {r: i for i, r in enumerate(self.routes)}
        self.vocab = json.loads((self.path / "vocab.json").read_text())
        if (
            set(self.vocab) != set(FIELDS)
            or [len(self.vocab[f]) + 3 for f in FIELDS] != m["vocab_sizes"]
        ):
            raise ValueError("artifact vocabulary sizes mismatch")
        for mapping in self.vocab.values():
            if sorted(mapping.values()) != list(range(3, len(mapping) + 3)):
                raise ValueError("artifact vocabulary IDs are not contiguous")
        scaling = json.loads((self.path / "scaling.json").read_text())
        if (
            scaling["scale"] != m["scale"]
            or scaling["start"] != m["start"]
            or scaling["end"] != m["end"]
        ):
            raise ValueError("scaling metadata mismatch")
        self.counts = np.load(
            self.path / "boardings.npy", mmap_mode="r", allow_pickle=False
        )
        self.offsets = np.load(
            self.path / "offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self.events = (
            np.load(self.path / "events.npy", mmap_mode="r", allow_pickle=False)
            if events
            else None
        )
        shape = (len(self.routes), (self.end - self.start).days * 24)
        if self.counts.shape != shape or self.offsets.shape != (*shape, 2):
            raise ValueError("artifact grid shape mismatch")
        if self.events is not None and (
            self.events.shape != (m["events"], 5) or self.events.dtype != np.int32
        ):
            raise ValueError("artifact event shape/dtype mismatch")
        if int(self.offsets[-1, -1, 1]) != m["events"]:
            raise ValueError("artifact terminal offset mismatch")
        flat = self.offsets.reshape(-1, 2)
        if (
            self.offsets.dtype != np.int64
            or self.counts.dtype != np.float32
            or flat[0, 0] != 0
            or (flat[:, 1] < flat[:, 0]).any()
            or not np.array_equal(flat[1:, 0], flat[:-1, 1])
        ):
            raise ValueError(
                "artifact offsets are not contiguous or dtypes are invalid"
            )
        if not np.isfinite(self.counts).all() or (self.counts < 0).any():
            raise ValueError("invalid artifact boarding values")
        self.contract = {
            "fingerprint": m["fingerprint"],
            "vocab": self.vocab,
            "scale": m["scale"],
            "routes": self.routes,
            "start": m["start"],
            "end": m["end"],
        }
        self.contract_hash = digest(self.contract)

    def history(self, route, cutoff, include_events=True, history_days=21):
        if type(history_days) is not int or history_days < 1:
            raise ValueError("history_days must be a positive integer")
        if route not in self.route_index:
            raise ValueError("unsupported route")
        stop = (cutoff - self.start).days * 24
        begin = stop - history_days * 24
        if begin < 0 or stop > self.counts.shape[1]:
            raise ValueError("history outside fitting artifact")
        r = self.route_index[route]
        counts = np.array(self.counts[r, begin:stop], copy=True)
        hours = None
        if include_events:
            if self.events is None:
                raise ValueError("event storage was not opened")
            hours = []
            for left, right in self.offsets[r, begin:stop]:
                if not 0 <= left <= right <= len(self.events):
                    raise ValueError("invalid event offsets")
                hours.append(
                    np.array(self.events[left:right], dtype=np.int64, copy=True)
                )
        return counts, hours

    def target(self, route, day):
        offset = (day - self.start).days * 24
        if not 0 <= offset <= self.counts.shape[1] - 24:
            raise ValueError("target outside fitting interval")
        return np.array(
            self.counts[self.route_index[route], offset : offset + 24], copy=True
        )

    def evaluation_targets(self):
        if self.metadata["regime"] != "validation":
            raise ValueError("final artifact has no observed future targets")
        values = np.load(
            self.path / "evaluation.npy", mmap_mode="r", allow_pickle=False
        )
        expected = (
            len(self.routes) + 1,
            (date.fromisoformat(self.metadata["evaluation_end"]) - self.end).days * 24,
        )
        if values.shape != expected:
            raise ValueError("evaluation target shape mismatch")
        return values
