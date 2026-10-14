"""Shared fixtures. NOTE: the synthetic arrays here exist only to unit-test code paths.

They are never used as a stand-in for PEMS04 experimental results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.data.pems04 import PEMS04Dataset

TINY_T, TINY_N, TINY_C = 600, 10, 3


def make_tiny_array(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(TINY_T)[:, None]
    s = np.arange(TINY_N)[None, :]
    flow = 100.0 * (s + 1) + 20.0 * np.sin(2 * np.pi * t / 288.0) + rng.normal(0, 3, size=(TINY_T, TINY_N))
    occ = 0.05 * (s + 1) + rng.normal(0, 0.005, size=(TINY_T, TINY_N))
    speed = 65.0 - 1.5 * s + rng.normal(0, 1, size=(TINY_T, TINY_N))
    return np.stack([flow, occ, speed], axis=-1).astype(np.float32)


@pytest.fixture
def tiny_array() -> np.ndarray:
    return make_tiny_array()


@pytest.fixture
def tiny_npz(tmp_path: Path, tiny_array: np.ndarray) -> Path:
    path = tmp_path / "raw" / "pems04.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, data=tiny_array)
    return path


@pytest.fixture
def tiny_dataset(tiny_npz: Path) -> PEMS04Dataset:
    return PEMS04Dataset(tiny_npz, url="http://invalid.local/pems04.npz", expected_shape=(TINY_T, TINY_N, TINY_C))
