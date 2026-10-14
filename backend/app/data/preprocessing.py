"""Chronological splitting, normalisation and memory-efficient window indexing.

Forecasting windows are never materialised for the whole dataset. A window is
identified by an integer pair ``(t, sensor)`` and gathered on demand from the
underlying ``(T, N, C)`` array:

    input  = data[t : t + lookback, sensor, :]              -> (lookback, C)
    target = data[t + lookback : t + lookback + horizon, sensor, flow]  -> (horizon,)

A window is only valid inside a split if ``t >= split_start`` and
``t + lookback + horizon <= split_end``; this guarantees that neither the input
window nor the target horizon crosses a split boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SplitBounds:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def chronological_split(
    num_timesteps: int, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2
) -> SplitBounds:
    """Split the time axis into contiguous, ordered train / val / test ranges."""
    if num_timesteps < 3:
        raise ValueError("Need at least 3 time steps to split.")
    if min(train_ratio, val_ratio, test_ratio) < 0 or abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must be non-negative and sum to 1.")
    t1 = int(round(num_timesteps * train_ratio))
    t2 = int(round(num_timesteps * (train_ratio + val_ratio)))
    return SplitBounds(train=(0, t1), val=(t1, t2), test=(t2, num_timesteps))


def window_starts(bounds: tuple[int, int], lookback: int, horizon: int) -> np.ndarray:
    """All window start indices whose input AND horizon lie fully inside ``bounds``."""
    start, end = bounds
    last_start = end - lookback - horizon  # inclusive
    if last_start < start:
        return np.zeros(0, dtype=np.int64)
    return np.arange(start, last_start + 1, dtype=np.int64)


def count_windows(bounds: tuple[int, int], lookback: int, horizon: int, num_sensors: int) -> int:
    return int(window_starts(bounds, lookback, horizon).size * num_sensors)


def sample_window_pairs(
    bounds: tuple[int, int],
    lookback: int,
    horizon: int,
    num_sensors: int,
    max_examples: int | None,
    seed: int,
    sensor_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Deterministically sample ``(t, sensor)`` window pairs without replacement.

    Returns ``(pairs, total_available)``. If ``max_examples`` is ``None`` or at least
    the number of available windows, all windows are returned.
    """
    starts = window_starts(bounds, lookback, horizon)
    sensors = np.arange(num_sensors, dtype=np.int64) if sensor_ids is None else np.asarray(sensor_ids, dtype=np.int64)
    total = int(starts.size * sensors.size)
    if total == 0:
        return np.zeros((0, 2), dtype=np.int64), 0
    if max_examples is None or max_examples >= total:
        tt, ss = np.meshgrid(starts, sensors, indexing="ij")
        pairs = np.stack([tt.ravel(), ss.ravel()], axis=1)
        return pairs, total
    rng = np.random.default_rng(seed)
    flat = rng.choice(total, size=int(max_examples), replace=False)
    t_idx = flat // sensors.size
    s_idx = flat % sensors.size
    pairs = np.stack([starts[t_idx], sensors[s_idx]], axis=1).astype(np.int64)
    return pairs, total


def pairs_within_bounds(pairs: np.ndarray, bounds: tuple[int, int], lookback: int, horizon: int) -> bool:
    """True when every window (input + horizon) lies strictly inside ``bounds``."""
    if pairs.size == 0:
        return True
    t = pairs[:, 0]
    return bool(np.all(t >= bounds[0]) and np.all(t + lookback + horizon <= bounds[1]))


def gather_windows(
    data: np.ndarray, pairs: np.ndarray, lookback: int, horizon: int, target_channel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised gather of a batch of windows from the ``(T, N, C)`` array."""
    t = pairs[:, 0]
    s = pairs[:, 1]
    in_idx = t[:, None] + np.arange(lookback)[None, :]
    out_idx = t[:, None] + lookback + np.arange(horizon)[None, :]
    x = data[in_idx, s[:, None], :]
    y = data[out_idx, s[:, None], target_channel]
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)


class FlowScaler:
    """Per-channel z-score normalisation fitted on the training period only."""

    def __init__(self, flow_channel: int = 0, eps: float = 1e-6) -> None:
        self.flow_channel = flow_channel
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray, train_bounds: tuple[int, int]) -> "FlowScaler":
        subset = data[train_bounds[0] : train_bounds[1]]
        self.mean = subset.mean(axis=(0, 1)).astype(np.float32)
        std = subset.std(axis=(0, 1)).astype(np.float32)
        self.std = np.where(std < self.eps, 1.0, std).astype(np.float32)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "Scaler not fitted"
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_flow(self, y_norm: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "Scaler not fitted"
        return y_norm * self.std[self.flow_channel] + self.mean[self.flow_channel]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "z-score per channel, statistics fitted on the training period only",
            "mean": [float(v) for v in (self.mean if self.mean is not None else [])],
            "std": [float(v) for v in (self.std if self.std is not None else [])],
        }


class WindowDataset:
    """Index-based dataset: stores only ``(t, sensor)`` pairs, builds windows on demand."""

    def __init__(self, data: np.ndarray, pairs: np.ndarray, lookback: int, horizon: int, target_channel: int = 0) -> None:
        self.data = data
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.lookback = lookback
        self.horizon = horizon
        self.target_channel = target_channel

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = gather_windows(self.data, self.pairs[index : index + 1], self.lookback, self.horizon, self.target_channel)
        return x[0], y[0]

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gather_windows(self.data, self.pairs[indices], self.lookback, self.horizon, self.target_channel)
