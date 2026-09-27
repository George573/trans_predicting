"""Small Python integration example; no HTTP server or legacy head export."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import torch

from .data import Boardings, ROUTES, history_inputs, request_inputs
from .events import ScheduledEvents
from .model import ForecastNetwork


@dataclass(frozen=True)
class ForecastContext:
    route: int
    cutoff: date
    _encoded: torch.Tensor = field(repr=False)
    _events: ScheduledEvents = field(repr=False)
    _owner: object = field(repr=False)


class InferenceRunner:
    """Encode a context once and reuse it for independently requested days.

    Create with InferenceRunner(load_model(checkpoint, device="cpu")).
    Treat the model and returned context as read-only during their lifetime.
    """

    def __init__(self, model: ForecastNetwork):
        self.model = model.eval()
        self._owner = object()

    @torch.inference_mode()
    def apply_context(self, boardings: Boardings, route: int, cutoff: date) -> ForecastContext:
        """Encode the model's history window before cutoff, without future labels."""
        if route not in ROUTES or route not in boardings.routes:
            raise ValueError(f"Unsupported route: {route}")
        device = next(self.model.parameters()).device
        inputs = history_inputs(
            boardings, route, cutoff,
            history_days=getattr(self.model, "history_days", 14),
        )
        history = {
            name: torch.as_tensor(values[None], dtype=torch.float32, device=device)
            for name, values in inputs.items()
        }
        encoded = self.model.encode_history(history)
        return ForecastContext(route, cutoff, encoded, boardings.scheduled_events, self._owner)

    @torch.inference_mode()
    def predict_day(self, context: ForecastContext, day: date) -> np.ndarray:
        """Return 24 float32 hourly counts for one day, at leads 1 through 61."""
        if context._owner is not self._owner:
            raise ValueError("Context belongs to a different runner")
        lead = (day - context.cutoff).days + 1
        if not 1 <= lead <= 61:
            raise ValueError("Target day must be within 61 days starting at the cutoff")
        device = context._encoded.device
        features = request_inputs(context._events, context.route, [day])
        values = self.model.predict_day(
            context._encoded,
            torch.tensor([ROUTES.index(context.route) + 1], dtype=torch.long, device=device),
            torch.as_tensor(features, dtype=torch.float32, device=device),
            torch.tensor([lead], dtype=torch.float32, device=device),
        )
        return values[0].cpu().numpy().copy()

    def predict(self, boardings: Boardings, route: int, cutoff: date, day: date) -> np.ndarray:
        """Convenience call: apply context and predict one day together."""
        return self.predict_day(self.apply_context(boardings, route, cutoff), day)
