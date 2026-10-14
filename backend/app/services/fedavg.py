"""Federated Averaging over PyTorch state dictionaries + payload serialisation."""
from __future__ import annotations

import io
from typing import Iterable, Mapping

import torch

StateDict = Mapping[str, torch.Tensor]


def clone_state_dict(state_dict: StateDict) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def serialize_state_dict(state_dict: StateDict) -> bytes:
    """Serialise a state dict with ``torch.save`` into an in-memory buffer.

    This is the documented payload format used for every communication-cost
    measurement in FedTraffic. It includes PyTorch's container overhead but no
    network-layer (TCP/HTTP/TLS) overhead.
    """
    buffer = io.BytesIO()
    torch.save(clone_state_dict(state_dict), buffer)
    return buffer.getvalue()


def state_dict_num_bytes(state_dict: StateDict) -> int:
    return len(serialize_state_dict(state_dict))


def raw_tensor_bytes(state_dict: StateDict) -> int:
    """Sum of ``numel * element_size`` over all tensors (container-free size)."""
    return int(sum(v.numel() * v.element_size() for v in state_dict.values()))


def parameter_count(state_dict: StateDict) -> int:
    return int(sum(v.numel() for v in state_dict.values()))


def fedavg_aggregate(state_dicts: list[StateDict], num_examples: Iterable[int]) -> dict[str, torch.Tensor]:
    """Weighted FedAvg:  W = sum_k (n_k / sum_j n_j) * W_k.

    Floating-point tensors (parameters and float buffers) are averaged with weights
    proportional to the number of local training examples. Integer buffers (such as
    BatchNorm's ``num_batches_tracked``) cannot be meaningfully averaged and are taken
    from the client with the most examples.
    """
    counts = [int(n) for n in num_examples]
    if not state_dicts:
        raise ValueError("No client state dicts to aggregate.")
    if len(counts) != len(state_dicts):
        raise ValueError("num_examples must have one entry per state dict.")
    if any(n < 0 for n in counts) or sum(counts) <= 0:
        raise ValueError("num_examples must be non-negative with a positive total.")
    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("All client state dicts must share the same keys.")
    total = float(sum(counts))
    weights = [n / total for n in counts]
    largest = max(range(len(counts)), key=lambda i: counts[i])
    aggregated: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = state_dicts[0][key]
        if reference.is_floating_point() or reference.is_complex():
            acc = torch.zeros_like(reference, dtype=torch.float64)
            for sd, w in zip(state_dicts, weights):
                if sd[key].shape != reference.shape:
                    raise ValueError(f"Shape mismatch for '{key}'.")
                acc += sd[key].detach().cpu().to(torch.float64) * w
            aggregated[key] = acc.to(reference.dtype)
        else:
            aggregated[key] = state_dicts[largest][key].detach().cpu().clone()
    return aggregated
