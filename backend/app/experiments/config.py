"""Experiment configuration schema and presets."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

THREE_SEEDS: list[int] = [42, 123, 2026]


class Method(str, Enum):
    centralized = "centralized"
    fed_iid = "fed_iid"
    fed_noniid = "fed_noniid"
    comparison = "comparison"


METHOD_LABELS: dict[str, str] = {
    "centralized": "Centralized baseline",
    "fed_iid": "FedAvg (IID clients)",
    "fed_noniid": "FedAvg (non-IID clients)",
    "comparison": "Complete comparison",
}


class ExperimentConfig(BaseModel):
    dataset: Literal["PEMS04"] = "PEMS04"
    method: Method = Method.comparison
    num_clients: int = Field(4, ge=3, le=5, description="Simulated clients (3-5)")
    rounds: int = Field(5, ge=1, le=500, description="Federated rounds (centralized: round = local_epochs epochs)")
    local_epochs: int = Field(1, ge=1, le=50)
    learning_rate: float = Field(1e-3, gt=0, le=1.0)
    batch_size: int = Field(64, ge=4, le=4096)
    seed_mode: Literal["single", "three_seed"] = "single"
    seed: int = Field(42, ge=0)
    model_type: Literal["gru"] = "gru"
    hidden_size: int = Field(32, ge=4, le=512)
    num_layers: int = Field(1, ge=1, le=4)
    lookback: int = Field(12, ge=1, le=288)
    horizon: int = Field(12, ge=1, le=288)
    max_train_examples: int | None = Field(4000, ge=50, description="Total training windows (None = all)")
    max_eval_examples: int | None = Field(2000, ge=50, description="Validation/test windows (None = all)")
    label: str | None = Field(None, max_length=120)

    @model_validator(mode="after")
    def _check_clients(self) -> "ExperimentConfig":
        if self.max_train_examples is not None and self.max_train_examples < self.num_clients * self.batch_size // 2:
            # keep each client with at least a few batches
            pass
        return self

    def seeds(self) -> list[int]:
        return list(THREE_SEEDS) if self.seed_mode == "three_seed" else [self.seed]

    def methods(self) -> list[str]:
        if self.method == Method.comparison:
            return [Method.centralized.value, Method.fed_iid.value, Method.fed_noniid.value]
        return [self.method.value]

    def plan(self) -> list[dict[str, Any]]:
        return [{"method": m, "seed": s} for s in self.seeds() for m in self.methods()]

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


QUICK_PRESET = ExperimentConfig(
    method=Method.comparison,
    num_clients=4,
    rounds=5,
    local_epochs=1,
    learning_rate=1e-3,
    batch_size=64,
    seed_mode="single",
    seed=42,
    hidden_size=32,
    max_train_examples=4000,
    max_eval_examples=2000,
    label="Quick comparison",
)

FULL_PRESET = ExperimentConfig(
    method=Method.comparison,
    num_clients=4,
    rounds=20,
    local_epochs=1,
    learning_rate=1e-3,
    batch_size=64,
    seed_mode="three_seed",
    seed=42,
    hidden_size=32,
    max_train_examples=20000,
    max_eval_examples=5000,
    label="Three-seed evaluation",
)


def presets() -> dict[str, Any]:
    return {
        "quick": QUICK_PRESET.dump(),
        "full": FULL_PRESET.dump(),
        "three_seeds": THREE_SEEDS,
        "methods": [{"value": k, "label": v} for k, v in METHOD_LABELS.items()],
        "limits": {
            "num_clients": {"min": 3, "max": 5},
            "rounds": {"min": 1, "max": 500},
            "local_epochs": {"min": 1, "max": 50},
            "batch_size": {"min": 4, "max": 4096},
            "hidden_size": {"min": 4, "max": 512},
        },
    }
