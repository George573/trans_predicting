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
        if model_kind == "full":
            self.events = EventEncoder(vocab_sizes, self.config)
            self.raw = MultiscaleConv1d(38, self.config.hourly)
        self.boarding = MultiscaleConv1d(7, self.config.hourly)
        self.shared = nn.Sequential(
            MultiscaleConv1d(64 if model_kind == "full" else 32, self.config.shared1),
            nn.MaxPool1d(2, 2),
            MultiscaleConv1d(64, self.config.shared2),
            nn.MaxPool1d(2, 2),
            MultiscaleConv1d(32, self.config.shared3),
            nn.MaxPool1d(3, 3),
        )
        self.route = nn.Embedding(10, 8)
        self.head = nn.Sequential(
            nn.Linear(685, 128),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 24),
        )
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(1)))
        total = sum(p.numel() for p in self.parameters())
        if not 100000 <= total <= 400000:
            raise ValueError(f"parameter budget exceeded: {total}")

    def encode_pooled_history(self, pooled, counts, calendar):
        """pooled [B,32,504], counts [B,1,504] in original units, calendar [B,6,504]."""
        b = counts.shape[0]
        if (
            counts.shape != (b, 1, 504)
            or calendar.shape != (b, 6, 504)
            or pooled.shape != (b, 32, 504)
        ):
            raise ValueError("history must cover exactly 504 aligned hours")
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
                counts.new_zeros((b, 32, 504)), counts, history["calendar"]
            )
        if "hours" in history:
            if len(history["hours"]) != b * 504:
                raise ValueError("ragged history must contain exactly B*504 hours")
            pooled = (
                self.events.encode_hours(history["hours"])
                .reshape(b, 504, 32)
                .transpose(1, 2)
            )
            return self.encode_pooled_history(pooled, counts, history["calendar"])
        indices = history["hour_indices"]
        if (
            indices.ndim != 1
            or indices.unique().numel() != indices.numel()
            or (indices < 0).any()
            or (indices >= b * 504).any()
        ):
            raise ValueError("hour indices must be unique and in range")
        pooled = counts.new_zeros((b * 504, 32))
        if indices.numel():
            encoded = self.events(history["event_ids"], history["lengths"])
            if encoded.shape[0] != indices.numel():
                raise ValueError("event/hour count mismatch")
            pooled = pooled.index_copy(0, indices, encoded)
        pooled = pooled.reshape(b, 504, 32).transpose(1, 2)
        return self.encode_pooled_history(pooled, counts, history["calendar"])

    def predict_day(self, encoded_history, route_indices, request_calendar, lead):
        b = encoded_history.shape[0]
        if (
            encoded_history.shape != (b, 672)
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
        return self.predict_day(
            self.encode_history(history),
            request["route_indices"],
            request["calendar"],
            request["lead"],
        )

    def parameter_report(self):
        report = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in self.named_children()
        }
        report["total"] = sum(report.values())
        return report
