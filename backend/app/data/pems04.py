"""Loader, validator and downloader for the real PEMS04 (PeMSD4) traffic dataset.

Source: https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04
Archive: pems04.npz -> NPZ key ``data`` with shape (16992, 307, 3)
         (time steps x sensors x [flow, occupancy, speed]), 5-minute resolution.

The NPZ does not contain timestamps or sensor coordinates. Timestamps shown by the
API are *derived* from the documented start date (January 2018) and the 5-minute
interval, and are only produced when the loaded array matches the documented
number of time steps. No geographic coordinates are invented.
"""
from __future__ import annotations

import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from app.data.preprocessing import chronological_split


class DatasetError(Exception):
    """Base class for dataset problems."""


class DatasetNotAvailableError(DatasetError):
    """Raised when the PEMS04 archive has not been downloaded yet."""


class DatasetValidationError(DatasetError):
    """Raised when an archive exists but does not look like PEMS04."""


FEATURES: list[dict[str, Any]] = [
    {"index": 0, "key": "flow", "name": "Traffic flow", "unit": "vehicles / 5 min", "role": "model input + forecast target"},
    {"index": 1, "key": "occupancy", "name": "Occupancy", "unit": "ratio (0-1)", "role": "model input"},
    {"index": 2, "key": "speed", "name": "Speed", "unit": "mph", "role": "model input"},
]

MANUAL_INSTRUCTIONS = (
    "Download pems04.npz from "
    "https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz "
    "(or from the ASTGCN GitHub repository, folder data/PEMS04) and place it at "
    "backend/data/raw/pems04.npz. Alternatively run `python scripts/download_pems04.py` "
    "inside the backend directory or press 'Download dataset' in the web UI."
)


@dataclass
class DownloadState:
    in_progress: bool = False
    bytes_downloaded: int = 0
    total_bytes: int | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.total_bytes:
            d["percent"] = round(100.0 * self.bytes_downloaded / self.total_bytes, 1)
        else:
            d["percent"] = None
        return d


def _py(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON serialisation."""
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class PEMS04Dataset:
    """Thread-safe access to the PEMS04 archive with lazy loading and caching."""

    name = "PEMS04"
    source_repository = "https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04"
    citation = (
        "Guo et al., 'Attention Based Spatial-Temporal Graph Convolutional Networks "
        "for Traffic Flow Forecasting', AAAI 2019 (PeMSD4 release)."
    )

    def __init__(
        self,
        path: Path,
        url: str,
        *,
        start_date: str | None = "2018-01-01",
        expected_shape: tuple[int, int, int] = (16992, 307, 3),
        sampling_interval_minutes: int = 5,
        flow_channel: int = 0,
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    ) -> None:
        self.path = Path(path)
        self.url = url
        self.start_date = start_date
        self.expected_shape = expected_shape
        self.sampling_interval_minutes = sampling_interval_minutes
        self.flow_channel = flow_channel
        self.split_ratios = split_ratios

        self._lock = threading.RLock()
        self._array: np.ndarray | None = None
        self._loaded_mtime: float | None = None
        self._load_error: str | None = None
        self._archive_info: dict[str, Any] = {}
        self._sensor_train_stats: tuple[np.ndarray, np.ndarray] | None = None
        self.download_state = DownloadState()

    # ------------------------------------------------------------------ files
    def file_exists(self) -> bool:
        return self.path.is_file()

    def file_size(self) -> int | None:
        return self.path.stat().st_size if self.file_exists() else None

    # ------------------------------------------------------------ inspection
    @staticmethod
    def inspect_archive(path: Path) -> dict[str, Any]:
        """Return the keys / shapes / dtypes stored in an NPZ archive."""
        try:
            with np.load(path, allow_pickle=False) as npz:
                keys = list(npz.files)
                arrays = {}
                for key in keys:
                    arr = npz[key]
                    arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        except (OSError, ValueError, EOFError) as exc:  # zip / pickle / truncated file
            raise DatasetValidationError(f"Cannot read NPZ archive: {exc}") from exc
        return {"keys": keys, "arrays": arrays}

    @staticmethod
    def select_array_key(info: dict[str, Any]) -> str:
        """Pick the array that holds the (T, N, C) traffic tensor."""
        if "data" in info["keys"]:
            return "data"
        for key in info["keys"]:
            if len(info["arrays"][key]["shape"]) == 3:
                return key
        raise DatasetValidationError(
            f"No 3-D array found in archive (keys: {info['keys']}). Expected key 'data' "
            "with shape (time_steps, sensors, features)."
        )

    def validate_array(self, arr: np.ndarray) -> dict[str, Any]:
        if arr.ndim != 3:
            raise DatasetValidationError(
                f"Expected a 3-D array (time, sensors, features) but got shape {arr.shape}."
            )
        if not np.issubdtype(arr.dtype, np.number):
            raise DatasetValidationError(f"Array dtype {arr.dtype} is not numeric.")
        t, n, c = arr.shape
        if t < 100 or n < 1 or c < 1:
            raise DatasetValidationError(f"Array shape {arr.shape} is too small to be PEMS04.")
        if self.flow_channel >= c:
            raise DatasetValidationError(
                f"Configured flow channel {self.flow_channel} does not exist (only {c} features)."
            )
        nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
        if nan_count > 0.5 * arr.size:
            raise DatasetValidationError("More than half of the dataset values are NaN.")
        return {"nan_count": nan_count}

    # --------------------------------------------------------------- loading
    def load(self, force: bool = False) -> np.ndarray:
        """Load (and cache) the dataset as float32 array of shape (T, N, C)."""
        with self._lock:
            if not self.file_exists():
                self._array = None
                raise DatasetNotAvailableError(
                    f"PEMS04 archive not found at {self.path}. {MANUAL_INSTRUCTIONS}"
                )
            mtime = self.path.stat().st_mtime
            if self._array is not None and not force and mtime == self._loaded_mtime:
                return self._array
            try:
                info = self.inspect_archive(self.path)
                key = self.select_array_key(info)
                with np.load(self.path, allow_pickle=False) as npz:
                    raw = npz[key]
                checks = self.validate_array(raw)
                arr = np.ascontiguousarray(raw, dtype=np.float32)
            except DatasetValidationError as exc:
                self._load_error = str(exc)
                self._array = None
                raise
            except Exception as exc:  # pragma: no cover - defensive
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._array = None
                raise DatasetValidationError(self._load_error) from exc
            self._array = arr
            self._loaded_mtime = mtime
            self._load_error = None
            self._sensor_train_stats = None
            self._archive_info = {
                **info,
                "array_key": key,
                "original_dtype": str(raw.dtype),
                "nan_count": checks["nan_count"],
                "zero_flow_count": int((arr[:, :, self.flow_channel] == 0).sum()),
            }
            return arr

    def is_loaded(self) -> bool:
        return self._array is not None

    def invalidate(self) -> None:
        with self._lock:
            self._array = None
            self._loaded_mtime = None
            self._sensor_train_stats = None

    # ------------------------------------------------------------ timestamps
    def timestamps_supported(self, num_timesteps: int | None = None) -> bool:
        if not self.start_date:
            return False
        if num_timesteps is None:
            num_timesteps = self._array.shape[0] if self._array is not None else None
        return num_timesteps == self.expected_shape[0]

    def timestamp(self, index: int, num_timesteps: int | None = None) -> str | None:
        if not self.timestamps_supported(num_timesteps):
            return None
        start = datetime.fromisoformat(self.start_date)  # type: ignore[arg-type]
        return (start + timedelta(minutes=self.sampling_interval_minutes * int(index))).isoformat()

    def timestamps(self, indices: np.ndarray, num_timesteps: int | None = None) -> list[str] | None:
        if not self.timestamps_supported(num_timesteps):
            return None
        start = datetime.fromisoformat(self.start_date)  # type: ignore[arg-type]
        step = timedelta(minutes=self.sampling_interval_minutes)
        return [(start + step * int(i)).isoformat() for i in indices]

    # ----------------------------------------------------------------- split
    def split_bounds(self, num_timesteps: int):
        return chronological_split(num_timesteps, *self.split_ratios)

    def split_info(self, num_timesteps: int) -> dict[str, Any]:
        bounds = self.split_bounds(num_timesteps)
        out: dict[str, Any] = {
            "strategy": "chronological (no shuffling before splitting)",
            "ratios": {"train": self.split_ratios[0], "val": self.split_ratios[1], "test": self.split_ratios[2]},
            "partitions": {},
        }
        for name, (start, end) in bounds.as_dict().items():
            out["partitions"][name] = {
                "start_index": start,
                "end_index": end,
                "num_timesteps": end - start,
                "start_timestamp": self.timestamp(start, num_timesteps),
                "end_timestamp": self.timestamp(max(end - 1, start), num_timesteps),
            }
        return out

    # ---------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        """Describe availability / dimensions. Never raises."""
        exists = self.file_exists()
        status: dict[str, Any] = {
            "name": self.name,
            "source": "PeMSD4 release from the ASTGCN repository (Davidham3/ASTGCN, data/PEMS04)",
            "source_repository": self.source_repository,
            "source_url": self.url,
            "citation": self.citation,
            "local_path": str(self.path),
            "file_exists": exists,
            "file_size_bytes": self.file_size(),
            "expected_shape": list(self.expected_shape),
            "sampling_interval_minutes": self.sampling_interval_minutes,
            "features": FEATURES,
            "flow_channel": self.flow_channel,
            "download": self.download_state.to_dict(),
            "manual_instructions": MANUAL_INSTRUCTIONS,
            "load_status": "not_downloaded",
            "error": None,
            "shape": None,
            "num_timesteps": None,
            "num_sensors": None,
            "num_features": None,
        }
        if not exists:
            return status
        try:
            arr = self.load()
        except DatasetError as exc:
            status["load_status"] = "error"
            status["error"] = str(exc)
            return status
        t, n, c = arr.shape
        status.update(
            {
                "load_status": "loaded",
                "shape": [t, n, c],
                "num_timesteps": t,
                "num_sensors": n,
                "num_features": c,
                "archive_keys": self._archive_info.get("keys"),
                "array_key": self._archive_info.get("array_key"),
                "original_dtype": self._archive_info.get("original_dtype"),
                "in_memory_dtype": str(arr.dtype),
                "in_memory_bytes": int(arr.nbytes),
                "shape_matches_documentation": list(arr.shape) == list(self.expected_shape),
                "missing_values": {
                    "nan_count": self._archive_info.get("nan_count", 0),
                    "zero_flow_count": self._archive_info.get("zero_flow_count", 0),
                    "note": "PEMS04 encodes missing observations as zeros rather than NaN.",
                },
                "time_range": self._time_range(t),
                "split": self.split_info(t),
                "duration_days": round(t * self.sampling_interval_minutes / (60 * 24), 2),
                "features": FEATURES[:c] if c <= len(FEATURES) else FEATURES
                + [{"index": i, "key": f"feature_{i}", "name": f"Feature {i}", "unit": "unknown", "role": "model input"} for i in range(len(FEATURES), c)],
            }
        )
        return status

    def _time_range(self, num_timesteps: int) -> dict[str, Any]:
        if self.timestamps_supported(num_timesteps):
            return {
                "derived": True,
                "start": self.timestamp(0, num_timesteps),
                "end": self.timestamp(num_timesteps - 1, num_timesteps),
                "basis": (
                    "Derived from the documented PeMSD4 collection period (January-February 2018, "
                    "59 days x 288 five-minute steps). The NPZ archive itself contains no timestamps."
                ),
            }
        return {
            "derived": False,
            "start": None,
            "end": None,
            "basis": "Loaded array does not match the documented length; timestamps are not derived.",
        }

    # -------------------------------------------------------------- download
    def start_download(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self.download_state.in_progress:
                return {"status": "in_progress", "download": self.download_state.to_dict()}
            if self.file_exists() and not force:
                return {"status": "already_present", "download": self.download_state.to_dict()}
            self.download_state = DownloadState(in_progress=True, started_at=time.time())
            thread = threading.Thread(target=self._download_worker, name="pems04-download", daemon=True)
            thread.start()
            return {"status": "started", "download": self.download_state.to_dict()}

    def _download_worker(self) -> None:
        try:
            self.download(progress=True)
            self.download_state.last_result = "completed"
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.download_state.error = f"{type(exc).__name__}: {exc}"
            self.download_state.last_result = "failed"
        finally:
            self.download_state.in_progress = False
            self.download_state.finished_at = time.time()

    def download(self, progress: bool = False, timeout: int = 60) -> Path:
        """Synchronously download the archive to ``self.path`` (atomic rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".part")
        request = urllib.request.Request(self.url, headers={"User-Agent": "FedTraffic/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_path, "wb") as fh:
                length = response.headers.get("Content-Length")
                self.download_state.total_bytes = int(length) if length else None
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        self.download_state.bytes_downloaded = downloaded
        except urllib.error.URLError as exc:
            tmp_path.unlink(missing_ok=True)
            raise DatasetError(f"Download failed: {exc}. {MANUAL_INSTRUCTIONS}") from exc
        # Validate before moving into place so a broken file never masquerades as PEMS04.
        try:
            info = self.inspect_archive(tmp_path)
            key = self.select_array_key(info)
            with np.load(tmp_path, allow_pickle=False) as npz:
                self.validate_array(npz[key])
        except DatasetValidationError:
            tmp_path.unlink(missing_ok=True)
            raise
        os.replace(tmp_path, self.path)
        self.invalidate()
        return self.path

    # --------------------------------------------------------------- sensors
    def _check_sensor(self, arr: np.ndarray, sensor_id: int) -> None:
        if sensor_id < 0 or sensor_id >= arr.shape[1]:
            raise ValueError(f"sensor_id must be between 0 and {arr.shape[1] - 1}.")

    def sensor_train_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-sensor mean / std of traffic flow computed on the training period only."""
        arr = self.load()
        with self._lock:
            if self._sensor_train_stats is None:
                start, end = self.split_bounds(arr.shape[0]).train
                flow = arr[start:end, :, self.flow_channel]
                self._sensor_train_stats = (flow.mean(axis=0), flow.std(axis=0))
            return self._sensor_train_stats

    def sensor_statistics(self, sensor_id: int) -> dict[str, Any]:
        arr = self.load()
        self._check_sensor(arr, sensor_id)
        series = arr[:, sensor_id, :]
        t = arr.shape[0]
        train_start, train_end = self.split_bounds(t).train
        per_feature = []
        for feat in FEATURES[: arr.shape[2]]:
            col = series[:, feat["index"]]
            per_feature.append(
                {
                    "key": feat["key"],
                    "name": feat["name"],
                    "unit": feat["unit"],
                    "mean": _py(col.mean()),
                    "std": _py(col.std()),
                    "min": _py(col.min()),
                    "max": _py(col.max()),
                    "nan_count": int(np.isnan(col).sum()),
                    "zero_count": int((col == 0).sum()),
                }
            )
        flow = series[:, self.flow_channel]
        return {
            "sensor_id": sensor_id,
            "num_timesteps": t,
            "features": per_feature,
            "train_period_flow": {
                "mean": _py(flow[train_start:train_end].mean()),
                "std": _py(flow[train_start:train_end].std()),
            },
            "coordinates_available": False,
            "coordinates_note": "The PEMS04 NPZ archive provides no sensor coordinates; none are shown.",
        }

    def sensor_series(
        self,
        sensor_id: int,
        start: int = 0,
        end: int | None = None,
        max_points: int = 2016,
    ) -> dict[str, Any]:
        arr = self.load()
        self._check_sensor(arr, sensor_id)
        t = arr.shape[0]
        start = max(0, min(int(start), t - 1))
        end = t if end is None else max(start + 1, min(int(end), t))
        length = end - start
        stride = max(1, math.ceil(length / max(1, max_points)))
        indices = np.arange(start, end, stride)
        payload: dict[str, Any] = {
            "sensor_id": sensor_id,
            "start": start,
            "end": end,
            "stride": stride,
            "downsampled": stride > 1,
            "num_points": int(indices.size),
            "sampling_interval_minutes": self.sampling_interval_minutes,
            "indices": indices.tolist(),
            "timestamps": self.timestamps(indices, t),
        }
        for feat in FEATURES[: arr.shape[2]]:
            payload[feat["key"]] = _py(arr[indices, sensor_id, feat["index"]])
        payload["statistics"] = self.sensor_statistics(sensor_id)
        return payload
