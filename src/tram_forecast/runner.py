"""Small Python integration example; no HTTP server or legacy head export."""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import torch

from .data import ROUTES, Boardings, history_inputs, request_inputs
from .events import ScheduledEvents
from .model import ForecastNetwork


@dataclass(frozen=True)
class ForecastContext:
    route: int
    cutoff: date
    _encoded: torch.Tensor = field(repr=False)
    _history: np.ndarray = field(repr=False)
    _events: ScheduledEvents | None = field(repr=False)
    _owner: object = field(repr=False)


class InferenceRunner:
    """Encode a context once and reuse it for independently requested days.

    Create with InferenceRunner(load_model(checkpoint, device="cpu")).
    Treat the model and returned context as read-only during their lifetime.
    """

    def __init__(self, model: ForecastNetwork):
        self.model = model.eval()
        self._owner = object()

    def _check_context(self, context):
        if not isinstance(context, ForecastContext):
            raise TypeError("Expected a ForecastContext from apply_context")
        if context._owner is not self._owner:
            raise ValueError("Context belongs to a different runner")

    @torch.inference_mode()
    def apply_context(self, boardings: Boardings, route: int, cutoff: date) -> ForecastContext:
        """Encode the 336 hours immediately before cutoff, without future labels."""
        if route not in ROUTES or route not in boardings.routes:
            raise ValueError(f"Unsupported route: {route}")
        device = next(self.model.parameters()).device
        inputs = history_inputs(boardings, route, cutoff, history_days=14, missing_zero=True)
        if not np.isfinite(inputs["counts"]).all() or (inputs["counts"] < 0).any():
            raise ValueError("Context history must have finite, nonnegative counts")
        history = {
            name: torch.as_tensor(values[None], dtype=torch.float32, device=device)
            for name, values in inputs.items()
        }
        encoded = self.model.encode_history(history)
        return ForecastContext(route, cutoff, encoded, inputs["counts"].copy(),
                               boardings.scheduled_events, self._owner)

    @torch.inference_mode()
    def predict_day(self, context: ForecastContext, day: date) -> np.ndarray:
        """Return 24 float32 hourly counts for a day on or after the cutoff."""
        self._check_context(context)
        lead = (day - context.cutoff).days + 1
        if lead < 1:
            raise ValueError("Target day must be on or after the cutoff")
        device = context._encoded.device
        features = request_inputs(context._events, context.route, [day], missing_zero=True)
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

    def update_context(self, context: ForecastContext, day: date, counts) -> ForecastContext:
        """Advance the context by one day using 24 observed or predicted counts."""
        self._check_context(context)
        if day != context.cutoff:
            raise ValueError("Updated day must equal the context cutoff")
        counts = np.asarray(counts, dtype=np.float32)
        if counts.shape != (24,) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Updated counts must be 24 finite, nonnegative values")
        history = Boardings(
            np.concatenate((context._history[:, 24:], counts[None]), axis=1),
            (context.route,), day - timedelta(days=13), 1.0, context._events,
        )
        return self.apply_context(history, context.route, day + timedelta(days=1))
