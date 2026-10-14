"""Local / centralized training and evaluation helpers built on the GRU model."""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from app.data.preprocessing import FlowScaler, gather_windows


class RunCancelled(Exception):
    """Raised inside training loops when a cancellation was requested."""


ProgressCallback = Callable[[int, int, int, int, float], None]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # optional; the prototype targets CPU
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = False) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_on_pairs(
    model: nn.Module,
    data: np.ndarray,
    pairs: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    target_channel: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    progress_cb: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
    report_every: int = 4,
) -> list[float]:
    """Train ``model`` in place on the given windows; returns mean MSE loss per epoch."""
    if pairs.shape[0] == 0:
        raise ValueError("Cannot train on an empty set of windows.")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    n = pairs.shape[0]
    num_batches = math.ceil(n / batch_size)
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        running = 0.0
        seen = 0
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x, y = gather_windows(data, pairs[idx], lookback, horizon, target_channel)
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt), yt)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(idx)
            seen += len(idx)
            if progress_cb is not None and (b % report_every == 0 or b == num_batches - 1):
                progress_cb(epoch + 1, epochs, b + 1, num_batches, running / max(seen, 1))
            if cancel_check is not None and cancel_check():
                raise RunCancelled("Training cancelled by user request.")
        epoch_losses.append(running / max(seen, 1))
    return epoch_losses


@torch.no_grad()
def evaluate_on_pairs(
    model: nn.Module,
    data: np.ndarray,
    pairs: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    target_channel: int,
    scaler: FlowScaler,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """MAE / RMSE in original flow units plus the normalised-space MSE."""
    model.to(device)
    model.eval()
    abs_sum = 0.0
    sq_sum = 0.0
    norm_sq_sum = 0.0
    count = 0
    for start in range(0, pairs.shape[0], batch_size):
        x, y = gather_windows(data, pairs[start : start + batch_size], lookback, horizon, target_channel)
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float64)
        y64 = y.astype(np.float64)
        norm_sq_sum += float(((pred - y64) ** 2).sum())
        diff = scaler.inverse_flow(pred) - scaler.inverse_flow(y64)
        abs_sum += float(np.abs(diff).sum())
        sq_sum += float((diff**2).sum())
        count += diff.size
    if count == 0:
        return {"mae": None, "rmse": None, "mse_normalized": None, "num_examples": 0}
    return {
        "mae": abs_sum / count,
        "rmse": math.sqrt(sq_sum / count),
        "mse_normalized": norm_sq_sum / count,
        "num_examples": int(pairs.shape[0]),
    }


@torch.no_grad()
def predict_pairs(
    model: nn.Module,
    data: np.ndarray,
    pairs: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    target_channel: int,
    scaler: FlowScaler,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (input_flow, actual, predicted) in original flow units."""
    model.to(device)
    model.eval()
    inputs, actuals, preds = [], [], []
    for start in range(0, pairs.shape[0], batch_size):
        x, y = gather_windows(data, pairs[start : start + batch_size], lookback, horizon, target_channel)
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        inputs.append(scaler.inverse_flow(x[:, :, target_channel]))
        actuals.append(scaler.inverse_flow(y))
        preds.append(scaler.inverse_flow(pred))
    if not preds:
        empty = np.zeros((0, horizon), dtype=np.float32)
        return np.zeros((0, lookback), dtype=np.float32), empty, empty
    return np.concatenate(inputs), np.concatenate(actuals), np.concatenate(preds)
