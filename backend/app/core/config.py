"""Application settings for the FedTraffic backend.

All configurable values can be overridden through environment variables with the
``FEDTRAFFIC_`` prefix (see ``backend/.env.example``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_prefix="FEDTRAFFIC_",
        extra="ignore",
    )

    # --- server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )

    # --- storage ----------------------------------------------------------
    data_dir: Path = BACKEND_DIR / "data"
    results_dir: Path = BACKEND_DIR / "results"

    # --- PEMS04 dataset ---------------------------------------------------
    pems04_url: str = (
        "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz"
    )
    pems04_filename: str = "pems04.npz"
    # The ASTGCN paper documents PeMSD4 as Jan-Feb 2018 (59 days x 288 steps = 16992).
    pems04_start_date: str = "2018-01-01"
    pems04_expected_timesteps: int = 16992
    pems04_expected_sensors: int = 307
    pems04_expected_features: int = 3
    sampling_interval_minutes: int = 5
    flow_channel: int = 0
    occupancy_channel: int = 1
    speed_channel: int = 2

    # --- chronological split ---------------------------------------------
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # Fixed seed used to sample validation / test windows so that every compared
    # method (and every training seed) is evaluated on the identical examples.
    eval_sample_seed: int = 7

    # --- compute ----------------------------------------------------------
    torch_threads: int = 0  # 0 -> leave PyTorch defaults

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def dataset_path(self) -> Path:
        return self.raw_dir / self.pems04_filename

    @property
    def experiments_dir(self) -> Path:
        return self.results_dir / "experiments"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
