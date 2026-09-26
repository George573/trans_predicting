"""Read-only artifact access with explicit fitting/evaluation separation."""

import json
from datetime import date
from pathlib import Path

import numpy as np

from .io import digest


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        m = self.metadata
        if m["schema_version"] != 2 or digest(m["identity"]) != m["fingerprint"]:
            raise ValueError("incompatible or corrupt artifact metadata; prepare boardings in a new directory")
        self.start = date.fromisoformat(m["start"])
        self.end = date.fromisoformat(m["end"])
        self.routes = tuple(m["routes"])
        self.route_index = {r: i for i, r in enumerate(self.routes)}
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
        shape = (len(self.routes), (self.end - self.start).days * 24)
        if self.counts.shape != shape or self.counts.dtype != np.float32:
            raise ValueError("artifact boarding grid shape/dtype mismatch")
        if not np.isfinite(self.counts).all() or (self.counts < 0).any():
            raise ValueError("invalid artifact boarding values")
        self.contract = {
            "fingerprint": m["fingerprint"],
            "scale": m["scale"],
            "routes": self.routes,
            "start": m["start"],
            "end": m["end"],
        }
        self.contract_hash = digest(self.contract)

    def history(self, route, cutoff, history_days=21):
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
        return counts

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
