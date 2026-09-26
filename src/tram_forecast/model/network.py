"""Complete default network. Inputs contain observed history only."""

import math

import torch
from torch import nn

from ..config import Config
from .blocks import MultiscaleConv1d
from .events import EventEncoder


class ForecastNetwork(nn.Module):
    def __init__(self, vocab_sizes, scale, config=None, model_kind="full"):
        super().__init__()
        self.config = config or Config()
        if model_kind not in ("full", "boarding_only"):
            raise ValueError("unknown model kind")
        self.model_kind = model_kind
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be finite and >=1")
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.history_hours = self.config.history_days * 24
        self.event_width = (
            sum(p[0] for p in (self.config.event1, self.config.event2)[self.config.event_depth - 1])
            if model_kind == "full" else 0
        )
        hourly_width = sum(p[0] for p in self.config.hourly)
        if model_kind == "full":
            self.events = EventEncoder(vocab_sizes, self.config)
            self.raw = MultiscaleConv1d(self.event_width + 6, self.config.hourly, self.config.hourly_stride)
        self.boarding = MultiscaleConv1d(7, self.config.hourly, self.config.hourly_stride)
        layers = []
        channels = hourly_width * (2 if model_kind == "full" else 1)
        length = (self.history_hours + self.config.hourly_stride - 1) // self.config.hourly_stride
        pool = nn.MaxPool1d if self.config.temporal_pool == "max" else nn.AvgPool1d
        stages = (self.config.shared1, self.config.shared2, self.config.shared3)
        for paths, factor in zip(stages[:self.config.shared_depth], (2, 2, 3)):
            layers.extend((MultiscaleConv1d(channels, paths), pool(factor, factor)))
            channels = sum(p[0] for p in paths)
            length //= factor
        if self.config.pooled_hours is not None:
            adaptive = nn.AdaptiveMaxPool1d if self.config.temporal_pool == "max" else nn.AdaptiveAvgPool1d
            layers.append(adaptive(self.config.pooled_hours))
            length = self.config.pooled_hours
        self.shared = nn.Sequential(*layers)
        self.encoded_width = channels * length
        self.route = nn.Embedding(10, 8)
        self.head = nn.Sequential(
            nn.Linear(self.encoded_width + 13, self.config.head_width),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_width, 24),
        )
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(1)))

    def encode_pooled_history(self, pooled, counts, calendar):
        """Encode aligned event, count and calendar histories."""
        b = counts.shape[0]
        if (
            counts.shape != (b, 1, self.history_hours)
            or calendar.shape != (b, 6, self.history_hours)
            or pooled.shape != (b, self.event_width, self.history_hours)
        ):
            raise ValueError(f"history must cover exactly {self.history_hours} aligned hours")
        a = (
            self.raw(torch.cat((pooled, calendar), dim=1))
            if self.model_kind == "full"
            else None
        )
        y = self.boarding(torch.cat((counts / self.scale, calendar), dim=1))
        return self.shared(torch.cat((a, y), dim=1) if a is not None else y).flatten(1)

    def encode_history(self, history):
        """Use ragged `hours` in flattened batch/hour order, or padded reference inputs."""
        counts = history["counts"]
        b = counts.shape[0]
        if self.model_kind == "boarding_only":
            return self.encode_pooled_history(
                counts.new_zeros((b, self.event_width, self.history_hours)), counts, history["calendar"]
            )
        if "hours" in history:
            if len(history["hours"]) != b * self.history_hours:
                raise ValueError(f"ragged history must contain exactly B*{self.history_hours} hours")
            pooled = (
                self.events.encode_hours(history["hours"])
                .reshape(b, self.history_hours, self.event_width)
                .transpose(1, 2)
            )
            return self.encode_pooled_history(pooled, counts, history["calendar"])
        indices = history["hour_indices"]
        if (
            indices.ndim != 1
            or indices.unique().numel() != indices.numel()
            or (indices < 0).any()
            or (indices >= b * self.history_hours).any()
        ):
            raise ValueError("hour indices must be unique and in range")
        pooled = counts.new_zeros((b * self.history_hours, self.event_width))
        if indices.numel():
            encoded = self.events(history["event_ids"], history["lengths"])
            if encoded.shape[0] != indices.numel():
                raise ValueError("event/hour count mismatch")
            pooled = pooled.index_copy(0, indices, encoded)
        pooled = pooled.reshape(b, self.history_hours, self.event_width).transpose(1, 2)
        return self.encode_pooled_history(pooled, counts, history["calendar"])

    def predict_day(self, encoded_history, route_indices, request_calendar, lead):
        b = encoded_history.shape[0]
        if (
            encoded_history.shape != (b, self.encoded_width)
            or route_indices.shape != (b,)
            or request_calendar.shape != (b, 4)
            or lead.shape != (b,)
        ):
            raise ValueError("invalid request tensor shapes")
        if (route_indices < 1).any() or (route_indices > 9).any():
            raise ValueError("unsupported neural route index")
        if (lead < 1).any() or (lead > 61).any() or (lead != lead.round()).any():
            raise ValueError("lead must be an integer in 1..61")
        x = torch.cat(
            (
                encoded_history,
                self.route(route_indices),
                request_calendar,
                ((lead - 1) / 60)[:, None],
            ),
            dim=1,
        )
        return (
            torch.nn.functional.softplus(self.head(x), beta=1, threshold=20)
            * self.scale
        )

    def forward(self, history, request):
        encoded = self.encode_history(history)
        indices = request.get("context_indices")
        if indices is not None:
            if (
                indices.ndim != 1
                or indices.dtype != torch.long
                or (indices < 0).any()
                or (indices >= encoded.shape[0]).any()
            ):
                raise ValueError("invalid request context indices")
            encoded = encoded.index_select(0, indices)
        return self.predict_day(
            encoded, request["route_indices"], request["calendar"],
            request["lead"],
        )

    def parameter_report(self):
        report = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in self.named_children()
        }
        report.setdefault("events", 0)
        report.setdefault("raw", 0)
        report["total"] = sum(report.values())
        return report
