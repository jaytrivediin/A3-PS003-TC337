"""Compact GRU traffic-flow forecaster (PyTorch)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GRUForecaster(nn.Module):
    """Many-to-one GRU: (batch, lookback, features) -> (batch, horizon) flow forecast."""

    def __init__(self, input_size: int = 3, hidden_size: int = 32, horizon: int = 12, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, lookback, features), got {tuple(x.shape)}")
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "GRU",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "horizon": self.horizon,
            "output_layer": "Linear(hidden_size -> horizon)",
            "parameter_count": count_parameters(self),
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(input_size: int, hidden_size: int, horizon: int, num_layers: int = 1) -> GRUForecaster:
    return GRUForecaster(input_size=input_size, hidden_size=hidden_size, horizon=horizon, num_layers=num_layers)
