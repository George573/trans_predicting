"""Training loop with a separate train/val DataLoader."""

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ForecastDataset, collate_samples
from .model import ForecastNetwork


def mae(prediction, target):
    return (prediction - target).abs().mean()


def train(
    train_boardings,
    val_boardings,
    forecast_days=7,
    epochs=30,
    batch_size=32,
    val_batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    clip_norm=1.0,
    device="cpu",
    output="checkpoint.pt",
):
    train_loader = DataLoader(
        ForecastDataset(train_boardings, forecast_days),
        batch_size=batch_size, shuffle=True, collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        ForecastDataset(val_boardings, forecast_days),
        batch_size=val_batch_size, shuffle=False, collate_fn=collate_samples,
    )

    model = ForecastNetwork(train_boardings.scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_wape = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for inputs, request, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            request = {k: v.to(device) for k, v in request.items()}
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = mae(model(inputs, request), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            total_loss += float(loss) * target.shape[0]
            seen += target.shape[0]

        model.eval()
        abs_error, actual_total = 0.0, 0.0
        with torch.no_grad():
            for inputs, request, target in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                request = {k: v.to(device) for k, v in request.items()}
                target = target.to(device)
                prediction = model(inputs, request)
                abs_error += float((prediction - target).abs().sum())
                actual_total += float(target.sum())
        wape = abs_error / actual_total if actual_total else float("inf")

        print(f"Epoch {epoch + 1}/{epochs} | MAE={total_loss / seen:.4f} | val WAPE={wape:.4%}")
        if wape < best_wape:
            best_wape = wape
            torch.save({"model": model.state_dict(), "scale": train_boardings.scale}, output)

    return output


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ForecastNetwork(payload["scale"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model