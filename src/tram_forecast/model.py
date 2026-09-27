"""A fixed CNN for 14 days of hourly boarding history and holiday features."""

import math

import torch
from torch import nn

from .data import HISTORY_CALENDAR_FEATURES, REQUEST_CALENDAR_FEATURES


class ParallelConv1d(nn.Module):
    """Several dilated Conv1d branches over the same input, concatenated by channel."""

    def __init__(self, params):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(in_c, out_c, k_size, dilation=dilation, padding="same")
            for in_c, out_c, k_size, dilation in params
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ForecastNetwork(nn.Module):
    def __init__(self, scale):
        super().__init__()
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be finite and >=1")
        self.register_buffer("scale", torch.tensor(float(scale)))

        self.encoder = nn.Sequential(
            nn.Conv1d(1 + len(HISTORY_CALENDAR_FEATURES), 40, kernel_size=50, padding="same"),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(40, 40, kernel_size=50, padding="same"),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(40, 20, kernel_size=50, padding="same"),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Flatten(1),  # 42 * 20 = 840
        )
        self.route = nn.Embedding(10, 8)
        self.head = nn.Sequential(
            nn.Linear(840 + 8 + len(REQUEST_CALENDAR_FEATURES) + 1, 320),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(320, 320),  # history, route, calendar, lead
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(320, 24),
        )
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(1)))

    def encode_history(self, history):
        counts, calendar = history["counts"], history["calendar"]
        return self.encoder(torch.cat((counts / self.scale, calendar), dim=1))

    def predict_day(self, encoded_history, route_indices, request_calendar, lead):
        x = torch.cat(
            (
                encoded_history,
                self.route(route_indices),
                request_calendar,
                ((lead - 1) / 60)[:, None],
            ),
            dim=1,
        )
        return torch.nn.functional.softplus(self.head(x), beta=1, threshold=20) * self.scale

    def forward(self, history, request):
        encoded = self.encode_history(history)
        indices = request.get("context_indices")
        if indices is not None:
            encoded = encoded.index_select(0, indices)
        return self.predict_day(
            encoded, request["route_indices"], request["calendar"], request["lead"],
        )

    def parameter_report(self):
        report = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in self.named_children()
        }
        report["total"] = sum(report.values())
        return report
