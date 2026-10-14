"""Dataset endpoints: status, download, sensor explorer, partition preview."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from app.data.partitioning import iid_partition, noniid_sensor_partition, partition_description
from app.data.pems04 import PEMS04Dataset
from app.data.preprocessing import count_windows, sample_window_pairs

router = APIRouter(prefix="/dataset", tags=["dataset"])


def _ds(request: Request) -> PEMS04Dataset:
    return request.app.state.dataset


@router.get("/status")
def dataset_status(request: Request):
    return _ds(request).status()


@router.post("/download")
def dataset_download(request: Request, force: bool = False):
    return _ds(request).start_download(force=force)


@router.get("/sensors")
def sensors_overview(request: Request):
    ds = _ds(request)
    arr = ds.load()
    mean, std = ds.sensor_train_statistics()
    return {
        "num_sensors": int(arr.shape[1]),
        "train_mean_flow": [round(float(v), 3) for v in mean],
        "train_std_flow": [round(float(v), 3) for v in std],
        "basis": "training period only",
    }


@router.get("/sensor/{sensor_id}")
def sensor_series(
    request: Request,
    sensor_id: int,
    start: int = Query(0, ge=0),
    end: int | None = Query(None, ge=1),
    max_points: int = Query(2016, ge=10, le=20000),
):
    try:
        return _ds(request).sensor_series(sensor_id, start=start, end=end, max_points=max_points)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/windows")
def window_counts(request: Request, lookback: int = Query(12, ge=1, le=288), horizon: int = Query(12, ge=1, le=288)):
    ds = _ds(request)
    arr = ds.load()
    bounds = ds.split_bounds(arr.shape[0])
    return {
        "lookback": lookback,
        "horizon": horizon,
        "windows": {name: count_windows(b, lookback, horizon, arr.shape[1]) for name, b in bounds.as_dict().items()},
    }


@router.get("/partitions/preview")
def partition_preview(
    request: Request,
    strategy: Literal["iid", "noniid"] = "noniid",
    num_clients: int = Query(4, ge=3, le=5),
    seed: int = Query(42, ge=0),
    max_train_examples: int | None = Query(4000, ge=50),
    lookback: int = Query(12, ge=1, le=288),
    horizon: int = Query(12, ge=1, le=288),
):
    ds = _ds(request)
    raw = ds.load()
    bounds = ds.split_bounds(raw.shape[0])
    mean, _ = ds.sensor_train_statistics()
    if strategy == "iid":
        pairs, _available = sample_window_pairs(bounds.train, lookback, horizon, raw.shape[1], max_train_examples, seed)
        clients = iid_partition(pairs, num_clients, seed, raw, bounds.train, ds.flow_channel)
    else:
        per_client = None if max_train_examples is None else max(1, max_train_examples // num_clients)
        clients = noniid_sensor_partition(raw, bounds.train, num_clients, per_client, lookback, horizon, seed, ds.flow_channel)
    assignment = [0] * raw.shape[1]
    if strategy == "noniid":
        for cl in clients:
            for s in cl.sensor_ids.tolist():
                assignment[int(s)] = cl.client_id
    return {
        "strategy": clients[0].strategy if clients else strategy,
        "description": partition_description(strategy),
        "num_clients": num_clients,
        "seed": seed,
        "clients": [cl.summary() for cl in clients],
        "sensor_train_mean_flow": [round(float(v), 2) for v in mean],
        "sensor_client_assignment": assignment if strategy == "noniid" else None,
        "basis": "training period only (validation/test data never used for partitioning)",
    }
