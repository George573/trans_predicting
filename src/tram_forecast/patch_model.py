"""Daily-patch attention with a local-scale, weekly seasonal residual head."""

import math

import torch
from torch import nn

from .data import HISTORY_INPUT_FEATURES, REQUEST_INPUT_FEATURES


class PatchForecastNetwork(nn.Module):
    """Encode observed days once; query any future day without autoregression.

    The packed 2-D context preserves the existing runner/fixed_forecast API.
    Counts are scaled using only the context, never future targets. A median
    weekly profile anchors the forecast before residual learning starts.
    """

    def __init__(self, scale=1.0, history_days=28, width=48, layers=2, dropout=0.1):
        super().__init__()
        if history_days < 7 or history_days % 7:
            raise ValueError("history_days must be a positive multiple of seven")
        if width % 4 or not math.isfinite(scale) or scale < 1:
            raise ValueError("width must be divisible by four and scale >= 1")
        self.config = dict(
            scale=float(scale),
            history_days=history_days,
            width=width,
            layers=layers,
            dropout=dropout,
        )
        self.history_days, self.width = history_days, width
        self.patch = nn.Linear(24 + len(HISTORY_INPUT_FEATURES), width)
        self.position = nn.Parameter(torch.randn(1, history_days, width) * 0.02)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            width * 2,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.route = nn.Embedding(10, 8)
        self.query = nn.Linear(len(REQUEST_INPUT_FEATURES) + 8 + 3, width)
        self.attention = nn.MultiheadAttention(
            width, 4, dropout=dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2 + 24 + 1, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 24),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode_history(self, history):
        counts = history["counts"]
        if counts.shape[-1] != self.history_days * 24:
            raise ValueError(f"expected {self.history_days * 24} history hours")
        daily = counts.reshape(-1, self.history_days, 24)
        local_scale = daily.mean((1, 2)).clamp_min(1)
        normalized = daily / local_scale[:, None, None]
        calendar = (
            history["calendar"]
            .reshape(
                -1,
                len(HISTORY_INPUT_FEATURES),
                self.history_days,
                24,
            )
            .mean(-1)
            .transpose(1, 2)
        )
        tokens = self.patch(torch.cat((normalized, calendar), -1)) + self.position
        tokens = self.norm(self.encoder(tokens))
        # torch.quantile interpolates the middle pair for an even number of weeks.
        weekly = normalized.reshape(-1, self.history_days // 7, 7, 24).quantile(
            0.5, dim=1
        )
        return torch.cat(
            (tokens.flatten(1), weekly.flatten(1), local_scale[:, None]), 1
        )

    def predict_day(self, encoded_history, route_indices, request_calendar, lead):
        split = self.history_days * self.width
        tokens = encoded_history[:, :split].reshape(-1, self.history_days, self.width)
        weekly = encoded_history[:, split:-1].reshape(-1, 7, 24)
        baseline = weekly[
            torch.arange(len(lead), device=lead.device), (lead.long() - 1) % 7
        ]
        local_scale = encoded_history[:, -1:]
        lead_features = torch.stack(
            (
                (lead - 1) / 60,
                torch.sin(lead / 7 * 2 * math.pi),
                torch.cos(lead / 7 * 2 * math.pi),
            ),
            -1,
        )
        query = self.query(
            torch.cat((request_calendar, self.route(route_indices), lead_features), -1)
        )
        attended, _ = self.attention(query[:, None], tokens, tokens, need_weights=False)
        residual = self.head(
            torch.cat((query, attended[:, 0], baseline, local_scale.log()), -1)
        )
        # Nonnegative counts and bounded corrections for stable long-horizon learning.
        return (
            torch.expm1((torch.log1p(baseline) + 2 * residual.tanh()).clamp_min(0))
            * local_scale
        )

    def forward(self, history, request):
        encoded = self.encode_history(history)
        if "context_indices" in request:
            encoded = encoded.index_select(0, request["context_indices"])
        return self.predict_day(
            encoded, request["route_indices"], request["calendar"], request["lead"]
        )
