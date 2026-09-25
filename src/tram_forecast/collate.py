"""Deterministic ragged packing; empty hours never enter the event CNN."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HourChunk:
    indices: tuple
    tiled: bool = False


def pack_hours(hours, max_hours, max_positions):
    if max_hours < 1 or max_positions < 1:
        raise ValueError("chunk limits must be positive")
    for hour in hours:
        if hour.ndim != 2 or hour.shape[1] != 5 or hour.dtype != torch.long:
            raise ValueError("each hour must be an int64 tensor [N,5]")
    ordered = sorted(
        (i for i, h in enumerate(hours) if len(h)), key=lambda i: (len(hours[i]), i)
    )
    pending = []
    for i in ordered:
        n = len(hours[i])
        if pending and (
            len(pending) == max_hours or (len(pending) + 1) * n > max_positions
        ):
            yield HourChunk(tuple(pending))
            pending = []
        if n > max_positions:
            yield HourChunk((i,), True)
        else:
            pending.append(i)
    if pending:
        yield HourChunk(tuple(pending))


def pad_hours(hours, indices, device):
    lengths = torch.tensor([len(hours[i]) for i in indices], device=device)
    ids = torch.zeros(
        len(indices), int(lengths.max()), 5, dtype=torch.long, device=device
    )
    for j, i in enumerate(indices):
        ids[j, : len(hours[i])] = hours[i].to(device)
    return ids, lengths
