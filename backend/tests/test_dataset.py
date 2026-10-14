from pathlib import Path

import numpy as np
import pytest

from app.data.pems04 import DatasetNotAvailableError, DatasetValidationError, PEMS04Dataset
from tests.conftest import TINY_C, TINY_N, TINY_T


def test_archive_loads_and_validates_shape(tiny_dataset: PEMS04Dataset):
    arr = tiny_dataset.load()
    assert arr.shape == (TINY_T, TINY_N, TINY_C)
    assert arr.dtype == np.float32
    status = tiny_dataset.status()
    assert status["load_status"] == "loaded"
    assert status["shape"] == [TINY_T, TINY_N, TINY_C]
    assert status["array_key"] == "data"
    assert status["num_sensors"] == TINY_N
    assert status["split"]["partitions"]["train"]["start_index"] == 0


def test_missing_file_is_reported_not_substituted(tmp_path: Path):
    ds = PEMS04Dataset(tmp_path / "missing.npz", url="http://invalid.local/x.npz")
    status = ds.status()
    assert status["load_status"] == "not_downloaded"
    assert status["file_exists"] is False
    assert "pems04.npz" in status["manual_instructions"]
    with pytest.raises(DatasetNotAvailableError):
        ds.load()


def test_invalid_archive_rejected(tmp_path: Path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, data=np.zeros((10, 5)))  # 2-D -> invalid
    ds = PEMS04Dataset(bad, url="http://invalid.local/x.npz")
    with pytest.raises(DatasetValidationError):
        ds.load()
    assert ds.status()["load_status"] == "error"

    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"this is not a zip archive")
    ds2 = PEMS04Dataset(garbage, url="http://invalid.local/x.npz")
    with pytest.raises(DatasetValidationError):
        ds2.load()


def test_fallback_to_first_3d_array(tmp_path: Path, tiny_array: np.ndarray):
    path = tmp_path / "other.npz"
    np.savez(path, meta=np.arange(3), traffic=tiny_array)
    ds = PEMS04Dataset(path, url="http://invalid.local/x.npz")
    assert ds.load().shape == tiny_array.shape
    assert ds.status()["array_key"] == "traffic"


def test_timestamps_only_when_documented_length_matches(tiny_npz: Path):
    matching = PEMS04Dataset(tiny_npz, url="u", expected_shape=(TINY_T, TINY_N, TINY_C))
    assert matching.status()["time_range"]["derived"] is True
    assert matching.timestamp(0) == "2018-01-01T00:00:00"
    assert matching.timestamp(12) == "2018-01-01T01:00:00"
    mismatching = PEMS04Dataset(tiny_npz, url="u", expected_shape=(16992, 307, 3))
    assert mismatching.status()["time_range"]["derived"] is False
    assert mismatching.timestamp(0) is None


def test_sensor_series_and_validation(tiny_dataset: PEMS04Dataset):
    series = tiny_dataset.sensor_series(2, start=0, end=300, max_points=100)
    assert series["sensor_id"] == 2
    assert series["num_points"] <= 100
    assert len(series["flow"]) == series["num_points"]
    assert series["statistics"]["coordinates_available"] is False
    with pytest.raises(ValueError):
        tiny_dataset.sensor_series(TINY_N + 5)
