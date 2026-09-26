"""Event embedding and masked convolution reference implementation."""

import torch
from torch import nn

from .blocks import MultiscaleConv1d


class EventEncoder(nn.Module):
    def __init__(self, vocab_sizes, config):
        super().__init__()
        if len(vocab_sizes) != 5 or any(
            type(v) is not int or v < 3 or v > cap + 3
            for v, cap in zip(vocab_sizes, config.category_caps)
        ):
            raise ValueError(
                "five vocabulary sizes, including three reserved IDs, must fit configured caps"
            )
        self.embeddings = nn.ModuleList(
            nn.Embedding(v, d, padding_idx=0)
            for v, d in zip(vocab_sizes, config.embedding_dims)
        )
        self.e1 = MultiscaleConv1d(sum(config.embedding_dims), config.event1)
        self.e2 = (MultiscaleConv1d(sum(p[0] for p in config.event1), config.event2)
                   if config.event_depth == 2 else None)
        self.width = sum(p[0] for p in (config.event1, config.event2)[config.event_depth - 1])
        self.config = config

    def features(self, ids, lengths):
        """ids [H,N,5], lengths [H]; includes zero-length reference rows."""
        if (
            ids.ndim != 3
            or ids.shape[2] != 5
            or ids.shape[1] < 1
            or lengths.shape != (ids.shape[0],)
        ):
            raise ValueError(
                "expected nonzero padded length and IDs [H,N,5], lengths [H]"
            )
        if (lengths < 0).any() or (lengths > ids.shape[1]).any():
            raise ValueError("invalid lengths")
        mask = (
            torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
        )[:, None, :]
        x = torch.cat(
            [embedding(ids[:, :, i]) for i, embedding in enumerate(self.embeddings)],
            dim=-1,
        ).transpose(1, 2)
        x = self.e1(x, mask)
        return (self.e2(x, mask) if self.e2 is not None else x), mask

    def forward(self, ids, lengths):
        features, mask = self.features(ids, lengths)
        if self.config.event_pool == "avg":
            return features.sum(dim=-1) / lengths.clamp_min(1)[:, None]
        # All-empty rows have a finite zero reduction; do not backprop through -inf.
        safe = mask | (lengths == 0)[:, None, None]
        return features.masked_fill(~safe, float("-inf")).max(dim=-1).values

    def _execute(self, function, *args):
        from torch.utils.checkpoint import checkpoint

        if self.training and torch.is_grad_enabled() and self.config.checkpoint_events:
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _tile(self, ids, start, stop):
        lengths = torch.tensor([ids.shape[1]], device=ids.device)
        features, _ = self.features(ids, lengths)
        core = features[:, :, start:stop]
        return core.sum(dim=-1) if self.config.event_pool == "avg" else core.max(dim=-1).values

    def encode_long_hour(self, hour):
        device = self.embeddings[0].weight.device
        halo = self.config.halo
        core = self.config.max_positions - 2 * halo
        maximum = None
        for start in range(0, len(hour), core):
            stop = min(start + core, len(hour))
            left = max(0, start - halo)
            right = min(len(hour), stop + halo)
            ids = hour[left:right].to(device).unsqueeze(0)
            value = self._execute(self._tile, ids, start - left, stop - left)
            # Strict comparison keeps the earliest tile on exact ties, including backward.
            maximum = (
                value
                if maximum is None
                else (maximum + value if self.config.event_pool == "avg"
                      else torch.where(value > maximum, value, maximum))
            )
        if maximum is None:
            return self.embeddings[0].weight.new_zeros((1, self.width))
        return maximum / len(hour) if self.config.event_pool == "avg" else maximum

    def encode_hours(self, hours):
        """Return [len(hours),32], preserving order; compact IDs may reside on CPU."""
        from ..collate import pack_hours, pad_hours

        device = self.embeddings[0].weight.device
        outputs = []
        positions = []
        for chunk in pack_hours(
            hours, self.config.max_hours, self.config.max_positions
        ):
            if chunk.tiled:
                values = self.encode_long_hour(hours[chunk.indices[0]])
            else:
                ids, lengths = pad_hours(hours, chunk.indices, device)
                values = self._execute(self.forward, ids, lengths)
            outputs.append(values)
            positions.extend(chunk.indices)
        result = self.embeddings[0].weight.new_zeros((len(hours), self.width))
        if outputs:
            result = result.index_copy(
                0, torch.tensor(positions, device=device), torch.cat(outputs)
            )
        return result
