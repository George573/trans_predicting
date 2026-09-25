"""MLP с эмбеддингами: route и час недели кодируются обучаемыми векторами.

Отличие от бустинга и от гармоник - в том, как задаётся профиль. Дерево нарезает каждую
ячейку route x hour отдельными разбиениями, гармоники приближают её суммой синусов, а сеть
раскладывает в произведение эмбеддингов: форма суток учится совместно по всем маршрутам и
получается гладкой. Для ансамбля важно именно это - ошибки лежат в других местах.

Прогноз считается в единицах среднего уровня маршрута, а лосс - L1 на сырых посадках,
то есть ровно числитель WAPE.
"""

from dataclasses import dataclass, asdict
from datetime import date

import numpy as np
import polars as pl
import torch
from torch import nn

from data_preparation import CYCLIC_FEATURES, CYCLIC_INTRAWEEK, CYCLIC_YEAR
from harmonic import daylight_hours

N_HOUR_OF_WEEK = 168
N_DAYTYPE = 3


@dataclass(frozen=True)
class MLPSpec:
    d_route: int = 8
    d_how: int = 16
    d_daytype: int = 4
    hidden: int = 256
    dropout: float = 0.05
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 2048
    n_seeds: int = 1
    daylight: bool = True
    cyclic: bool = False        # sin/cos часа, дня недели и дня месяца
    cyclic_year: bool = False   # плюс sin/cos дня года и месяца (вне диапазона для ноя-дек)
    device: str = "auto"

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _encode(df: pl.DataFrame, spec: MLPSpec, route_to_idx: dict[int, int]) -> dict[str, torch.Tensor]:
    routes = df["route"].to_numpy()
    idx = np.array([route_to_idx.get(int(r), -1) for r in routes], dtype=np.int64)

    hour = df["hour"].to_numpy().astype(np.int64)
    weekday = df["weekday"].to_numpy().astype(np.int64)
    how = (weekday - 1) * 24 + hour

    is_weekend = df["is_weekend"].to_numpy()
    is_holiday = df["is_holiday"].to_numpy()
    daytype = np.where(is_holiday, 2, np.where(is_weekend, 1, 0)).astype(np.int64)

    numeric = [df["is_short_working_day"].to_numpy().astype(np.float32)]
    if spec.daylight:
        numeric.append((daylight_hours(df["day_of_year"].to_numpy().astype(float)) / 24.0).astype(np.float32))
    if spec.cyclic:
        numeric.extend(df[c].to_numpy().astype(np.float32) for c in CYCLIC_INTRAWEEK)
    if spec.cyclic_year:
        numeric.extend(df[c].to_numpy().astype(np.float32) for c in CYCLIC_YEAR)

    return {
        "route": torch.from_numpy(np.clip(idx, 0, None)),
        "known": torch.from_numpy((idx >= 0).astype(np.float32)),
        "how": torch.from_numpy(how),
        "daytype": torch.from_numpy(daytype),
        "numeric": torch.from_numpy(np.stack(numeric, axis=1)),
    }


class ProfileMLP(nn.Module):
    def __init__(self, n_routes: int, n_numeric: int, spec: MLPSpec):
        super().__init__()
        self.emb_route = nn.Embedding(n_routes, spec.d_route)
        self.emb_how = nn.Embedding(N_HOUR_OF_WEEK, spec.d_how)
        self.emb_daytype = nn.Embedding(N_DAYTYPE, spec.d_daytype)
        width = spec.d_route + spec.d_how + spec.d_daytype + n_numeric
        self.net = nn.Sequential(
            nn.Linear(width, spec.hidden), nn.GELU(), nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden, spec.hidden // 2), nn.GELU(),
            nn.Linear(spec.hidden // 2, 1),
        )

    def forward(self, route, how, daytype, numeric):
        x = torch.cat([self.emb_route(route), self.emb_how(how), self.emb_daytype(daytype), numeric], dim=1)
        return nn.functional.softplus(self.net(x)).squeeze(-1)


def fit_mlp(train_df: pl.DataFrame, spec: MLPSpec, cut: date | None = None) -> dict:
    """обучает n_seeds сетей; прогноз потом усредняется по сидам"""
    device = resolve_device(spec.device)

    active = (
        train_df.group_by("route").agg(pl.col("target").sum().alias("total"))
        .filter(pl.col("total") > 0).sort("route")["route"].to_list()
    )
    route_to_idx = {int(r): i for i, r in enumerate(active)}

    scale = (
        train_df.filter(pl.col("route").is_in(active))
        .group_by("route").agg(pl.col("target").mean().alias("scale"))
    )
    scale_by_idx = np.ones(len(active), dtype=np.float32)
    for r, s in zip(scale["route"], scale["scale"]):
        scale_by_idx[route_to_idx[int(r)]] = max(float(s), 1.0)

    part = train_df.filter(pl.col("route").is_in(active))
    enc = _encode(part, spec, route_to_idx)
    y = torch.from_numpy(part["target"].to_numpy().astype(np.float32)).to(device)
    scale_t = torch.from_numpy(scale_by_idx).to(device)
    enc = {k: v.to(device) for k, v in enc.items()}
    n = y.shape[0]

    models = []
    for seed in range(spec.n_seeds):
        torch.manual_seed(seed)
        model = ProfileMLP(len(active), enc["numeric"].shape[1], spec).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)

        model.train()
        for _ in range(spec.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, spec.batch_size):
                b = perm[start:start + spec.batch_size]
                out = model(enc["route"][b], enc["how"][b], enc["daytype"][b], enc["numeric"][b])
                loss = torch.abs(out * scale_t[enc["route"][b]] - y[b]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        models.append(model)

    return {"models": models, "route_to_idx": route_to_idx, "scale": scale_by_idx, "spec": spec, "device": str(device)}


@torch.no_grad()
def predict_mlp(state: dict, df: pl.DataFrame) -> np.ndarray:
    """прогноз в порядке строк df; маршруты без обучения получают нули"""
    spec: MLPSpec = state["spec"]
    device = resolve_device(spec.device)
    enc = {k: v.to(device) for k, v in _encode(df, spec, state["route_to_idx"]).items()}
    scale_t = torch.from_numpy(state["scale"]).to(device)

    total = torch.zeros(df.height, device=device)
    for model in state["models"]:
        total += model(enc["route"], enc["how"], enc["daytype"], enc["numeric"]) * scale_t[enc["route"]]

    pred = (total / len(state["models"])) * enc["known"]
    return pred.clamp(min=0).cpu().numpy().astype(float)


def save_mlp(state: dict, path: str) -> None:
    """артефакт для инференса: веса всех сидов плюс всё, что нужно для кодирования признаков"""
    torch.save({
        "state_dicts": [m.state_dict() for m in state["models"]],
        "route_to_idx": state["route_to_idx"],
        "scale": state["scale"],
        "spec": state["spec"].as_dict(),
        "n_numeric": state["models"][0].net[0].in_features
                     - state["spec"].d_route - state["spec"].d_how - state["spec"].d_daytype,
    }, path)


def load_mlp(path: str, device: str = "auto") -> dict:
    blob = torch.load(path, weights_only=False)
    spec = MLPSpec(**blob["spec"])
    dev = resolve_device(device)
    models = []
    for sd in blob["state_dicts"]:
        model = ProfileMLP(len(blob["route_to_idx"]), blob["n_numeric"], spec).to(dev)
        model.load_state_dict(sd)
        model.eval()
        models.append(model)
    return {"models": models, "route_to_idx": blob["route_to_idx"], "scale": blob["scale"],
            "spec": spec, "device": str(dev)}
