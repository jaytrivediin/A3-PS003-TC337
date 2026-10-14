"""Simulated-client data partitioning strategies (IID and sensor-based non-IID).

Both strategies only ever look at the *training* period. The validation and test
periods are never used to build partitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.data.preprocessing import sample_window_pairs

IID_DESCRIPTION = (
    "IID: the sampled training windows (all sensors, training period only) are shuffled "
    "with the run seed and split into equally sized disjoint shards, one per client."
)
NONIID_DESCRIPTION = (
    "Non-IID (sensor-based): sensors are ranked by their training-period mean traffic flow "
    "and divided into contiguous groups; each client owns one group of sensors and samples "
    "its training windows exclusively from those sensors, so clients see different flow "
    "distributions (low-flow roads vs. high-flow roads)."
)


@dataclass
class ClientPartition:
    client_id: int
    pairs: np.ndarray
    sensor_ids: np.ndarray
    strategy: str
    mean_flow: float
    std_flow: float
    sensor_group_rank: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_examples(self) -> int:
        return int(self.pairs.shape[0])

    def summary(self, preview: int = 40) -> dict[str, Any]:
        ids = [int(s) for s in np.sort(self.sensor_ids)]
        return {
            "client_id": self.client_id,
            "name": f"Client {self.client_id}",
            "num_examples": self.num_examples,
            "num_sensors": len(ids),
            "sensor_ids_preview": ids[:preview],
            "sensor_ids_truncated": len(ids) > preview,
            "mean_flow": round(float(self.mean_flow), 3),
            "std_flow": round(float(self.std_flow), 3),
            "strategy": self.strategy,
            **self.extra,
        }


def sensor_train_statistics(data: np.ndarray, train_bounds: tuple[int, int], flow_channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of raw traffic flow per sensor, using the training period only."""
    flow = data[train_bounds[0] : train_bounds[1], :, flow_channel]
    return flow.mean(axis=0), flow.std(axis=0)


def _example_weighted_stats(pairs: np.ndarray, sensor_mean: np.ndarray, sensor_std: np.ndarray) -> tuple[float, float]:
    if pairs.shape[0] == 0:
        return 0.0, 0.0
    sensors = pairs[:, 1]
    return float(sensor_mean[sensors].mean()), float(sensor_std[sensors].mean())


def iid_partition(
    pairs: np.ndarray,
    num_clients: int,
    seed: int,
    data: np.ndarray,
    train_bounds: tuple[int, int],
    flow_channel: int = 0,
) -> list[ClientPartition]:
    """Shuffle the pooled training windows and split them into disjoint shards."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if pairs.shape[0] < num_clients:
        raise ValueError("Not enough training examples for the requested number of clients.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pairs.shape[0])
    shards = np.array_split(pairs[perm], num_clients)
    sensor_mean, sensor_std = sensor_train_statistics(data, train_bounds, flow_channel)
    clients = []
    for k, shard in enumerate(shards, start=1):
        mean_flow, std_flow = _example_weighted_stats(shard, sensor_mean, sensor_std)
        clients.append(
            ClientPartition(
                client_id=k,
                pairs=np.ascontiguousarray(shard),
                sensor_ids=np.unique(shard[:, 1]),
                strategy="iid",
                mean_flow=mean_flow,
                std_flow=std_flow,
            )
        )
    return clients


def noniid_sensor_partition(
    data: np.ndarray,
    train_bounds: tuple[int, int],
    num_clients: int,
    examples_per_client: int | None,
    lookback: int,
    horizon: int,
    seed: int,
    flow_channel: int = 0,
) -> list[ClientPartition]:
    """Assign disjoint sensor groups (ranked by training-period mean flow) to clients."""
    num_sensors = data.shape[1]
    if num_clients < 1 or num_clients > num_sensors:
        raise ValueError("num_clients must be between 1 and the number of sensors.")
    sensor_mean, sensor_std = sensor_train_statistics(data, train_bounds, flow_channel)
    order = np.argsort(sensor_mean, kind="stable")
    groups = np.array_split(order, num_clients)
    clients = []
    for k, group in enumerate(groups, start=1):
        group_sorted = np.sort(group)
        shard, available = sample_window_pairs(
            train_bounds, lookback, horizon, num_sensors, examples_per_client, seed * 101 + k, sensor_ids=group_sorted
        )
        mean_flow, std_flow = _example_weighted_stats(shard, sensor_mean, sensor_std)
        clients.append(
            ClientPartition(
                client_id=k,
                pairs=shard,
                sensor_ids=group_sorted,
                strategy="noniid_sensor",
                mean_flow=mean_flow,
                std_flow=std_flow,
                sensor_group_rank=k,
                extra={
                    "windows_available": int(available),
                    "sensor_mean_flow_min": round(float(sensor_mean[group].min()), 3),
                    "sensor_mean_flow_max": round(float(sensor_mean[group].max()), 3),
                },
            )
        )
    return clients


def partition_description(strategy: str) -> str:
    return IID_DESCRIPTION if strategy == "iid" else NONIID_DESCRIPTION


def has_duplicate_examples(clients: list[ClientPartition]) -> bool:
    """True if any (t, sensor) window appears in more than one client."""
    if not clients:
        return False
    all_pairs = np.concatenate([c.pairs for c in clients], axis=0)
    if all_pairs.shape[0] == 0:
        return False
    keys = all_pairs[:, 0].astype(np.int64) * (all_pairs[:, 1].max() + 1) + all_pairs[:, 1]
    return bool(np.unique(keys).size != keys.size)
