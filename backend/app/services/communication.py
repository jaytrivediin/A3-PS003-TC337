"""Communication-cost accounting for federated rounds and the centralized baseline.

All federated numbers are *measured* byte lengths of serialised model state
dictionaries (see ``fedavg.serialize_state_dict``). The centralized number is an
*estimate* of the raw numerical payload that would have to be uploaded once to a
central server. Neither figure includes network-layer overhead.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MEASUREMENT_NOTE = (
    "Bytes are the length of each model state_dict serialised with torch.save into an "
    "in-memory buffer (download: server -> client, upload: client -> server). They exclude "
    "TCP/HTTP/TLS overhead and are not packet captures."
)
CENTRALIZED_NOTE = (
    "Estimated one-time raw-data payload: the selected training windows (lookback x features "
    "inputs plus horizon targets) encoded as float32, as if every window were uploaded to the "
    "central server. This is an estimate derived from the actual selected arrays, not a "
    "measured network transfer, and the centralized baseline has no per-round model traffic."
)


def round_communication(
    round_index: int,
    download_bytes: int,
    upload_bytes_per_client: list[int],
    previous_cumulative: int,
    parameter_count: int,
    raw_parameter_bytes: int,
) -> dict[str, Any]:
    num_clients = len(upload_bytes_per_client)
    total_download = int(download_bytes) * num_clients
    total_upload = int(sum(upload_bytes_per_client))
    round_bytes = total_download + total_upload
    return {
        "round": round_index,
        "num_clients": num_clients,
        "download_bytes_per_client": [int(download_bytes)] * num_clients,
        "upload_bytes_per_client": [int(b) for b in upload_bytes_per_client],
        "bytes_per_client": [int(download_bytes) + int(b) for b in upload_bytes_per_client],
        "total_download_bytes": total_download,
        "total_upload_bytes": total_upload,
        "round_bytes": round_bytes,
        "cumulative_bytes": int(previous_cumulative) + round_bytes,
        "model_parameter_count": int(parameter_count),
        "raw_parameter_bytes": int(raw_parameter_bytes),
    }


def summarize_federated(rounds: list[dict[str, Any]], payload_bytes: int, num_clients: int) -> dict[str, Any]:
    total_download = int(sum(r["total_download_bytes"] for r in rounds))
    total_upload = int(sum(r["total_upload_bytes"] for r in rounds))
    return {
        "type": "federated",
        "rounds": len(rounds),
        "num_clients": num_clients,
        "model_payload_bytes_per_transfer": int(payload_bytes),
        "total_download_bytes": total_download,
        "total_upload_bytes": total_upload,
        "total_bytes": total_download + total_upload,
        "bytes_per_round_mean": (total_download + total_upload) / len(rounds) if rounds else 0,
        "measurement": "measured_model_payload",
        "note": MEASUREMENT_NOTE,
    }


def unique_covered_cells(pairs: np.ndarray, lookback: int, horizon: int, num_timesteps: int, num_sensors: int) -> int:
    """Number of distinct (time step, sensor) observations covered by the windows."""
    if pairs.shape[0] == 0:
        return 0
    span = lookback + horizon
    diff = np.zeros((num_timesteps + span + 1, num_sensors), dtype=np.int32)
    np.add.at(diff, (pairs[:, 0], pairs[:, 1]), 1)
    np.add.at(diff, (pairs[:, 0] + span, pairs[:, 1]), -1)
    coverage = np.cumsum(diff, axis=0) > 0
    return int(coverage.sum())


def estimate_centralized_payload(
    pairs: np.ndarray,
    lookback: int,
    horizon: int,
    num_channels: int,
    num_timesteps: int,
    num_sensors: int,
    itemsize: int = 4,
) -> dict[str, Any]:
    num_examples = int(pairs.shape[0])
    values_per_example = lookback * num_channels + horizon
    per_example_bytes = values_per_example * itemsize
    unique_cells = unique_covered_cells(pairs, lookback, horizon, num_timesteps, num_sensors)
    return {
        "type": "centralized",
        "measurement": "estimated_raw_data_payload",
        "num_examples": num_examples,
        "values_per_example": values_per_example,
        "itemsize_bytes": itemsize,
        "estimated_raw_data_upload_bytes": num_examples * per_example_bytes,
        "deduplicated_series_upload_bytes": unique_cells * num_channels * itemsize,
        "unique_observations_covered": unique_cells,
        "measured_model_transfer_bytes": 0,
        "per_round_bytes": 0,
        "total_bytes": num_examples * per_example_bytes,
        "note": CENTRALIZED_NOTE,
    }
