"""Original-unit objective and requested-day-weighted accumulation."""

import torch


def mae(prediction, target):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 24
    ):
        raise ValueError("loss expects matching [requests,24] predictions/targets")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite regression values")
    return (prediction - target).abs().mean()


def normalize_gradients(parameters, sample_count):
    if sample_count < 1:
        raise ValueError("empty accumulation group")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
