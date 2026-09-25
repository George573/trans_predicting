"""Atomic checkpoints and epoch-boundary reproducibility."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .io import digest
from .model import ForecastNetwork
from .settings import Settings


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot resume on a CPU-only host")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_checkpoint(path, model, store, settings, optimizer=None, **progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model": model.state_dict(),
        "model_kind": model.model_kind,
        "settings": settings.to_dict(),
        "artifact_contract": store.contract,
        "artifact_hash": store.contract_hash,
        "vocab_sizes": store.metadata["vocab_sizes"],
        "scale": store.metadata["scale"],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": rng_state(),
        "progress": progress,
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
    }
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    if digest(payload["artifact_contract"]) != payload["artifact_hash"]:
        raise ValueError("checkpoint artifact contract is corrupt")
    return payload


def load_model(path, store, device="cpu"):
    payload = read_checkpoint(path)
    if payload["artifact_hash"] != store.contract_hash:
        raise ValueError("checkpoint/artifact contract mismatch")
    settings = Settings.from_dict(payload["settings"])
    model = ForecastNetwork(
        payload["vocab_sizes"], payload["scale"], settings.model, payload["model_kind"]
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
