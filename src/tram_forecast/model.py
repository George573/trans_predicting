"""A fixed CNN for 21 days of hourly boarding history."""

import math

import torch
from torch import nn

from .config import HISTORY_DAYS


class ForecastNetwork(nn.Module):
    history_hours = HISTORY_DAYS * 24
    encoded_width = 16 * 42
    model_kind = "boarding_only"

    def __init__(self, scale):
        super().__init__()
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be finite and >=1")
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.encoder = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),                 # 504 -> 252 hours
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),                 # 252 -> 126 hours
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(3),                 # 126 -> 42 hours
            nn.Flatten(1),                   # 16 * 42 = 672 features
        )
        self.route = nn.Embedding(10, 8)
        self.head = nn.Sequential(
            nn.Linear(672 + 8 + 4 + 1, 250),  # history, route, calendar, lead
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(250, 24),
        )
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(1)))

    def encode_history(self, history):
        counts, calendar = history["counts"], history["calendar"]
        b = counts.shape[0]
        if counts.shape != (b, 1, 504) or calendar.shape != (b, 6, 504):
            raise ValueError("history must cover exactly 504 aligned hours")
        return self.encoder(torch.cat((counts / self.scale, calendar), dim=1))

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
