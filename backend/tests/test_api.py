import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import TINY_C, TINY_N, TINY_T, make_tiny_array  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        pems04_expected_timesteps=TINY_T,
        pems04_expected_sensors=TINY_N,
        pems04_expected_features=TINY_C,
        pems04_url="http://invalid.local/pems04.npz",
    )


@pytest.fixture
def client_without_data(tmp_path: Path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_with_data(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.raw_dir.mkdir(parents=True)
    np.savez(settings.dataset_path, data=make_tiny_array())
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, settings


def test_health(client_without_data):
    r = client_without_data.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_missing_dataset_is_reported(client_without_data):
    status = client_without_data.get("/api/dataset/status").json()
    assert status["load_status"] == "not_downloaded"
    assert status["file_exists"] is False
    r = client_without_data.get("/api/dataset/sensor/0")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "dataset_not_available"
    r = client_without_data.post("/api/experiments", json={"method": "centralized", "rounds": 1})
    assert r.status_code == 409


def test_dataset_status_and_sensor(client_with_data):
    client, _ = client_with_data
    status = client.get("/api/dataset/status").json()
    assert status["load_status"] == "loaded"
    assert status["shape"] == [TINY_T, TINY_N, TINY_C]
    series = client.get("/api/dataset/sensor/1", params={"start": 0, "end": 288}).json()
    assert series["num_points"] == 288 and len(series["flow"]) == 288
    assert client.get("/api/dataset/sensor/999").status_code == 404
    preview = client.get("/api/dataset/partitions/preview", params={"strategy": "noniid", "num_clients": 3, "max_train_examples": 300}).json()
    assert len(preview["clients"]) == 3
    assert sum(1 for a in preview["sensor_client_assignment"] if a == 1) >= 3


def test_invalid_experiment_config_rejected(client_with_data):
    client, _ = client_with_data
    assert client.post("/api/experiments", json={"num_clients": 10}).status_code == 422
    assert client.post("/api/experiments", json={"learning_rate": -1}).status_code == 422


def test_experiment_end_to_end_and_export(client_with_data):
    client, settings = client_with_data
    config = {
        "method": "comparison",
        "num_clients": 3,
        "rounds": 2,
        "local_epochs": 1,
        "learning_rate": 0.005,
        "batch_size": 32,
        "seed_mode": "single",
        "seed": 42,
        "max_train_examples": 240,
        "max_eval_examples": 60,
        "label": "api-test",
    }
    r = client.post("/api/experiments", json=config)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert len(job["runs"]) == 3
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job['job_id']}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.3)
    assert job["status"] == "completed", json.dumps(job, indent=1)[:2000]

    listing = client.get("/api/experiments").json()
    assert listing["count"] == 3
    by_method = {s["method"]: s for s in listing["experiments"]}
    assert set(by_method) == {"centralized", "fed_iid", "fed_noniid"}

    fed = client.get(f"/api/experiments/{by_method['fed_noniid']['experiment_id']}").json()
    assert fed["status"] == "completed"
    assert len(fed["history"]) == 2
    assert fed["communication"]["type"] == "federated"
    assert fed["communication"]["total_bytes"] == sum(h["round_bytes"] for h in fed["history"])
    assert fed["history"][0]["total_download_bytes"] == 3 * fed["model"]["state_dict_bytes"]
    assert fed["test_metrics"]["mae"] > 0
    assert fed["partitioning"]["strategy"] == "noniid_sensor"
    assert len(fed["partitioning"]["clients"]) == 3
    assert fed["data"]["train_sampled"] is True

    cen = client.get(f"/api/experiments/{by_method['centralized']['experiment_id']}/results").json()
    assert cen["communication"]["measurement"] == "estimated_raw_data_payload"
    assert cen["communication"]["measured_model_transfer_bytes"] == 0
    assert cen["test_metrics"]["rmse"] >= cen["test_metrics"]["mae"]
    # independent runs: centralized and federated metrics are not copies of each other
    assert cen["test_metrics"]["mae"] != fed["test_metrics"]["mae"]

    # persisted JSON on disk
    saved = json.loads((settings.experiments_dir / f"{fed['experiment_id']}.json").read_text())
    assert saved["experiment_id"] == fed["experiment_id"]
    assert (settings.models_dir / f"{fed['experiment_id']}.pt").is_file()

    export = client.get(f"/api/experiments/{fed['experiment_id']}/export")
    assert export.status_code == 200 and export.headers["content-type"].startswith("text/csv")
    assert export.text.count("\n") >= 3
    assert client.get("/api/export/experiments.csv").status_code == 200

    preds = client.get(f"/api/experiments/{fed['experiment_id']}/predictions", params={"sensor_id": 1, "steps": 20}).json()
    assert len(preds["actual"]) == len(preds["predicted"]) == 20
    assert preds["indices"][0] >= fed["dataset"]["split"]["partitions"]["test"]["start_index"]
