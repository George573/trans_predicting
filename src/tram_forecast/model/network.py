"""Complete default network. Inputs contain observed history only."""

import math

import torch
from torch import nn

from ..config import Config
from .blocks import MultiscaleConv1d


class ForecastNetwork(nn.Module):
    def __init__(self, scale, config=None, model_kind="boarding_only"):
        super().__init__()
        self.config = config or Config()
        if model_kind != "boarding_only":
            raise ValueError("unknown model kind")
        self.model_kind = model_kind
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be finite and >=1")
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.history_hours = self.config.history_days * 24
        hourly_width = sum(p[0] for p in self.config.hourly)
        self.boarding = MultiscaleConv1d(7, self.config.hourly, self.config.hourly_stride)
        layers = []
        channels = hourly_width
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

    def encode_history(self, history):
        """Encode bounded boarding counts and aligned calendar features."""
        counts, calendar = history["counts"], history["calendar"]
        b = counts.shape[0]
        if counts.shape != (b, 1, self.history_hours) or calendar.shape != (b, 6, self.history_hours):
            raise ValueError(f"history must cover exactly {self.history_hours} aligned hours")
        y = self.boarding(torch.cat((counts / self.scale, calendar), dim=1))
        return self.shared(y).flatten(1)

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
        report["total"] = sum(report.values())
        return report
