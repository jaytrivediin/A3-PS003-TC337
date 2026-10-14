"""Executes experiments: centralized GRU baseline and simulated FedAvg training.

Every run trains a real PyTorch model on windows drawn from the real PEMS04 array.
Progress is pushed through ``RunContext.report`` so that the web UI can render the
actual state of the simulated server / clients while training is happening.
"""
from __future__ import annotations

import copy
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.core.config import Settings
from app.data.partitioning import ClientPartition, iid_partition, noniid_sensor_partition, partition_description
from app.data.pems04 import PEMS04Dataset
from app.data.preprocessing import FlowScaler, SplitBounds, sample_window_pairs
from app.experiments.config import METHOD_LABELS, ExperimentConfig
from app.experiments.jobs import RunContext, utc_now
from app.experiments.store import ExperimentStore
from app.models.gru import GRUForecaster, count_parameters
from app.services.communication import estimate_centralized_payload, round_communication, summarize_federated
from app.services.fedavg import clone_state_dict, fedavg_aggregate, raw_tensor_bytes, state_dict_num_bytes
from app.services.training import (
    RunCancelled,
    evaluate_on_pairs,
    get_device,
    predict_pairs,
    set_global_seed,
    train_on_pairs,
)
from app.utils.metrics import mae as mae_metric
from app.utils.metrics import rmse as rmse_metric

WORKFLOW_STEPS = [
    {"key": "initialize", "label": "Server initializes the global model"},
    {"key": "distribute", "label": "Server distributes global parameters"},
    {"key": "local_training", "label": "Clients train local models"},
    {"key": "local_update", "label": "Clients produce updated parameters"},
    {"key": "upload", "label": "Updates are returned to the server"},
    {"key": "aggregate", "label": "Server applies FedAvg"},
    {"key": "evaluate", "label": "New global model is evaluated"},
    {"key": "next_round", "label": "Next round begins"},
]


@dataclass
class PreparedData:
    raw: np.ndarray
    data_norm: np.ndarray
    scaler: FlowScaler
    split: SplitBounds
    train_pairs: np.ndarray
    train_available: int
    val_pairs: np.ndarray
    val_available: int
    test_pairs: np.ndarray
    test_available: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data_norm.shape)  # type: ignore[return-value]


class ExperimentRunner:
    def __init__(self, dataset: PEMS04Dataset, settings: Settings, store: ExperimentStore) -> None:
        self.dataset = dataset
        self.settings = settings
        self.store = store
        self.device = get_device(prefer_gpu=False)
        self._norm_cache: tuple[int, np.ndarray, np.ndarray, FlowScaler, SplitBounds] | None = None
        self._lock = threading.Lock()
        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)

    # ------------------------------------------------------------- preparing
    def normalized(self) -> tuple[np.ndarray, np.ndarray, FlowScaler, SplitBounds]:
        raw = self.dataset.load()
        with self._lock:
            if self._norm_cache is not None and self._norm_cache[0] == id(raw):
                _, raw_c, norm_c, scaler_c, split_c = self._norm_cache
                return raw_c, norm_c, scaler_c, split_c
            split = self.dataset.split_bounds(raw.shape[0])
            scaler = FlowScaler(flow_channel=self.settings.flow_channel).fit(raw, split.train)
            data_norm = scaler.transform(raw)
            self._norm_cache = (id(raw), raw, data_norm, scaler, split)
            return raw, data_norm, scaler, split

    def prepare(self, cfg: ExperimentConfig, seed: int) -> PreparedData:
        raw, data_norm, scaler, split = self.normalized()
        _, num_sensors, _ = data_norm.shape
        lookback, horizon = cfg.lookback, cfg.horizon
        train_pairs, train_avail = sample_window_pairs(split.train, lookback, horizon, num_sensors, cfg.max_train_examples, seed)
        val_pairs, val_avail = sample_window_pairs(split.val, lookback, horizon, num_sensors, cfg.max_eval_examples, self.settings.eval_sample_seed)
        test_pairs, test_avail = sample_window_pairs(split.test, lookback, horizon, num_sensors, cfg.max_eval_examples, self.settings.eval_sample_seed + 1)
        if train_pairs.shape[0] == 0 or val_pairs.shape[0] == 0 or test_pairs.shape[0] == 0:
            raise ValueError("The chronological splits are too short for the requested lookback/horizon.")
        return PreparedData(raw, data_norm, scaler, split, train_pairs, train_avail, val_pairs, val_avail, test_pairs, test_avail)

    def build_model(self, cfg: ExperimentConfig, num_channels: int) -> GRUForecaster:
        return GRUForecaster(input_size=num_channels, hidden_size=cfg.hidden_size, horizon=cfg.horizon, num_layers=cfg.num_layers)

    # ------------------------------------------------------------------ run
    def run(self, ctx: RunContext) -> dict[str, Any]:
        cfg, method, seed = ctx.config, ctx.method, ctx.seed
        record = self.store.load(ctx.experiment_id) or {}
        record.update(
            {
                "experiment_id": ctx.experiment_id,
                "job_id": ctx.job_id,
                "label": cfg.label,
                "status": "running",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": seed,
                "seed_mode": cfg.seed_mode,
                "config": cfg.dump(),
                "started_at": utc_now(),
                "history": [],
                "error": None,
                "device": str(self.device),
            }
        )
        record.setdefault("created_at", utc_now())
        self.store.save(record)
        started = time.time()
        ctx.report(
            {
                "phase": "preparing",
                "message": "Loading PEMS04, fitting normalisation on the training period and sampling windows",
                "percent": 0,
                "current_round": 0,
                "total_rounds": cfg.rounds,
                "server": {"status": "initializing"},
                "clients": [],
                "history": [],
            }
        )
        try:
            set_global_seed(seed)
            prep = self.prepare(cfg, seed)
            record["dataset"] = self._dataset_section(prep)
            record["data"] = self._data_section(cfg, prep)
            if method == "centralized":
                result = self._run_centralized(ctx, cfg, seed, prep)
            elif method == "fed_iid":
                result = self._run_federated(ctx, cfg, seed, prep, "iid")
            elif method == "fed_noniid":
                result = self._run_federated(ctx, cfg, seed, prep, "noniid")
            else:
                raise ValueError(f"Unknown method '{method}'")
            record.update(result)
            record["status"] = "completed"
        except RunCancelled as exc:
            record["status"] = "cancelled"
            record["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - captured into the record for the UI
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        record["finished_at"] = utc_now()
        record["duration_seconds"] = round(time.time() - started, 2)
        self.store.save(record)
        return record

    # ------------------------------------------------------------ sections
    def _dataset_section(self, prep: PreparedData) -> dict[str, Any]:
        t, n, c = prep.shape
        return {
            "name": self.dataset.name,
            "source_url": self.dataset.url,
            "shape": [t, n, c],
            "sampling_interval_minutes": self.dataset.sampling_interval_minutes,
            "split": self.dataset.split_info(t),
            "normalization": prep.scaler.to_dict(),
            "preprocessing_note": (
                "Normalisation statistics are computed from the training period of the whole dataset and "
                "shared by all simulated clients. This is a benchmarking convenience, not a privacy mechanism."
            ),
        }

    def _data_section(self, cfg: ExperimentConfig, prep: PreparedData) -> dict[str, Any]:
        return {
            "train_examples": int(prep.train_pairs.shape[0]),
            "train_examples_available": int(prep.train_available),
            "train_sampled": int(prep.train_pairs.shape[0]) < int(prep.train_available),
            "val_examples": int(prep.val_pairs.shape[0]),
            "val_examples_available": int(prep.val_available),
            "test_examples": int(prep.test_pairs.shape[0]),
            "test_examples_available": int(prep.test_available),
            "eval_sampled": int(prep.test_pairs.shape[0]) < int(prep.test_available),
            "eval_sample_seed": self.settings.eval_sample_seed,
            "sampling_note": (
                "Windows are sampled deterministically without replacement. Validation and test windows use a "
                "fixed seed so every compared method and every training seed is evaluated on identical examples."
            ),
        }

    def _model_section(self, model: GRUForecaster, cfg: ExperimentConfig) -> dict[str, Any]:
        sd = model.state_dict()
        return {
            **model.describe(),
            "lookback": cfg.lookback,
            "input_minutes": cfg.lookback * self.dataset.sampling_interval_minutes,
            "horizon_minutes": cfg.horizon * self.dataset.sampling_interval_minutes,
            "state_dict_bytes": state_dict_num_bytes(sd),
            "raw_parameter_bytes": raw_tensor_bytes(sd),
            "loss": "MSE on z-scored flow",
            "optimizer": "Adam (re-initialised for every local training phase)",
        }

    def _prediction_samples(self, model: GRUForecaster, cfg: ExperimentConfig, prep: PreparedData, count: int = 6) -> list[dict[str, Any]]:
        pairs = prep.test_pairs[:count]
        inputs, actual, pred = predict_pairs(
            model, prep.data_norm, pairs, lookback=cfg.lookback, horizon=cfg.horizon,
            target_channel=self.settings.flow_channel, scaler=prep.scaler, device=self.device,
        )
        t_total = prep.shape[0]
        samples = []
        for i, (t, sensor) in enumerate(pairs.tolist()):
            samples.append(
                {
                    "sensor_id": int(sensor),
                    "window_start": int(t),
                    "forecast_origin": int(t + cfg.lookback),
                    "forecast_origin_timestamp": self.dataset.timestamp(t + cfg.lookback, t_total),
                    "input_flow": [round(float(v), 2) for v in inputs[i]],
                    "actual": [round(float(v), 2) for v in actual[i]],
                    "predicted": [round(float(v), 2) for v in pred[i]],
                    "mae": round(mae_metric(actual[i], pred[i]), 3),
                }
            )
        return samples

    def _save_model(self, model: GRUForecaster, experiment_id: str) -> str:
        path = self.store.model_path(experiment_id)
        torch.save(clone_state_dict(model.state_dict()), path)
        return str(path)

    def _evaluate(self, model: GRUForecaster, cfg: ExperimentConfig, prep: PreparedData, pairs: np.ndarray) -> dict[str, Any]:
        return evaluate_on_pairs(
            model, prep.data_norm, pairs, lookback=cfg.lookback, horizon=cfg.horizon,
            target_channel=self.settings.flow_channel, scaler=prep.scaler, device=self.device,
        )

    # ------------------------------------------------------- centralized run
    def _run_centralized(self, ctx: RunContext, cfg: ExperimentConfig, seed: int, prep: PreparedData) -> dict[str, Any]:
        t, n, c = prep.shape
        model = self.build_model(cfg, c)
        model_info = self._model_section(model, cfg)
        history: list[dict[str, Any]] = []
        state: dict[str, Any] = {
            "phase": "training",
            "message": "Centralized training on the pooled training windows",
            "percent": 0,
            "current_round": 0,
            "total_rounds": cfg.rounds,
            "server": {"status": "training", "round": 0, "epoch": 0, "total_epochs": cfg.local_epochs, "train_loss": None, "val_mae": None, "val_rmse": None, "parameter_count": model_info["parameter_count"]},
            "clients": [],
            "history": history,
        }

        def emit(force: bool = True) -> None:
            ctx.report(copy.deepcopy(state), force=force)

        emit()
        for r in range(1, cfg.rounds + 1):
            if ctx.cancelled():
                raise RunCancelled("Cancelled before round %d" % r)
            round_start = time.time()
            state["current_round"] = r
            state["server"]["status"] = "training"
            state["server"]["round"] = r

            def progress(epoch: int, epochs: int, batch: int, batches: int, loss: float, _r: int = r) -> None:
                state["server"].update({"epoch": epoch, "batch": batch, "total_batches": batches, "train_loss": round(loss, 5)})
                state["percent"] = round(100.0 * ((_r - 1) + ((epoch - 1) + batch / batches) / epochs) / cfg.rounds, 1)
                emit(force=False)

            losses = train_on_pairs(
                model, prep.data_norm, prep.train_pairs, lookback=cfg.lookback, horizon=cfg.horizon,
                target_channel=self.settings.flow_channel, epochs=cfg.local_epochs, learning_rate=cfg.learning_rate,
                batch_size=cfg.batch_size, seed=seed * 1000 + r, device=self.device, progress_cb=progress,
                cancel_check=ctx.cancelled,
            )
            state["server"]["status"] = "evaluating"
            emit()
            val = self._evaluate(model, cfg, prep, prep.val_pairs)
            entry = {
                "round": r,
                "epochs_completed": r * cfg.local_epochs,
                "epoch_losses": [round(l, 6) for l in losses],
                "train_loss_mean": round(float(np.mean(losses)), 6),
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_mse_normalized": val["mse_normalized"],
                "num_clients": 0,
                "total_download_bytes": 0,
                "total_upload_bytes": 0,
                "round_bytes": 0,
                "cumulative_bytes": 0,
                "duration_seconds": round(time.time() - round_start, 3),
                "timestamp": utc_now(),
            }
            history.append(entry)
            state["server"].update({"val_mae": val["mae"], "val_rmse": val["rmse"], "status": "round_complete"})
            state["percent"] = round(100.0 * r / cfg.rounds, 1)
            emit()

        state["phase"] = "evaluating"
        state["message"] = "Evaluating on the chronological test period"
        emit()
        test = self._evaluate(model, cfg, prep, prep.test_pairs)
        estimate = estimate_centralized_payload(prep.train_pairs, cfg.lookback, cfg.horizon, c, t, n)
        best = min(history, key=lambda h: h["val_mae"]) if history else None
        model_path = self._save_model(model, ctx.experiment_id)
        state["phase"] = "completed"
        state["server"]["status"] = "completed"
        state["percent"] = 100
        emit()
        return {
            "model": model_info,
            "partitioning": {
                "strategy": "none",
                "description": "Centralized baseline: all selected training windows are pooled on one server.",
                "clients": [],
            },
            "history": history,
            "test_metrics": {"mae": test["mae"], "rmse": test["rmse"], "num_examples": test["num_examples"]},
            "best_validation": {"round": best["round"], "val_mae": best["val_mae"], "val_rmse": best["val_rmse"]} if best else None,
            "communication": estimate,
            "predictions_sample": self._prediction_samples(model, cfg, prep),
            "model_path": model_path,
            "round_definition": f"One centralized round = {cfg.local_epochs} epoch(s) over the pooled training windows (same epoch budget as the federated clients).",
        }

    # --------------------------------------------------------- federated run
    def _partition(self, cfg: ExperimentConfig, seed: int, prep: PreparedData, strategy: str) -> list[ClientPartition]:
        if strategy == "iid":
            return iid_partition(prep.train_pairs, cfg.num_clients, seed, prep.raw, prep.split.train, self.settings.flow_channel)
        per_client = None if cfg.max_train_examples is None else max(1, cfg.max_train_examples // cfg.num_clients)
        return noniid_sensor_partition(
            prep.raw, prep.split.train, cfg.num_clients, per_client, cfg.lookback, cfg.horizon, seed, self.settings.flow_channel
        )

    def _run_federated(self, ctx: RunContext, cfg: ExperimentConfig, seed: int, prep: PreparedData, strategy: str) -> dict[str, Any]:
        t, n, c = prep.shape
        clients = self._partition(cfg, seed, prep, strategy)
        total_examples = int(sum(cl.num_examples for cl in clients))

        global_model = self.build_model(cfg, c)
        model_info = self._model_section(global_model, cfg)
        payload_bytes = model_info["state_dict_bytes"]
        param_count = model_info["parameter_count"]
        raw_bytes = model_info["raw_parameter_bytes"]

        history: list[dict[str, Any]] = []
        client_states: list[dict[str, Any]] = [
            {
                "client_id": cl.client_id,
                "name": f"Client {cl.client_id}",
                "num_examples": cl.num_examples,
                "num_sensors": int(cl.sensor_ids.size),
                "mean_flow": round(cl.mean_flow, 2),
                "status": "waiting",
                "epoch": 0,
                "total_epochs": cfg.local_epochs,
                "batch": 0,
                "total_batches": 0,
                "train_loss": None,
                "download_bytes": 0,
                "upload_bytes": 0,
                "upload_status": "pending",
            }
            for cl in clients
        ]
        state: dict[str, Any] = {
            "phase": "training",
            "message": f"FedAvg with {cfg.num_clients} simulated {('IID' if strategy == 'iid' else 'non-IID')} clients",
            "percent": 0,
            "current_round": 0,
            "total_rounds": cfg.rounds,
            "workflow_step": "initialize",
            "server": {
                "status": "initializing",
                "round": 0,
                "val_mae": None,
                "val_rmse": None,
                "aggregation": "FedAvg weighted by local example count",
                "parameter_count": param_count,
                "payload_bytes": payload_bytes,
                "cumulative_bytes": 0,
            },
            "clients": client_states,
            "history": history,
        }

        def emit(force: bool = True) -> None:
            ctx.report(copy.deepcopy(state), force=force)

        emit()
        cumulative = 0
        num_clients = len(clients)
        for r in range(1, cfg.rounds + 1):
            if ctx.cancelled():
                raise RunCancelled("Cancelled before round %d" % r)
            round_start = time.time()
            state["current_round"] = r
            state["server"].update({"round": r, "status": "distributing"})
            state["workflow_step"] = "distribute"
            for cs in client_states:
                cs.update({"status": "waiting", "epoch": 0, "batch": 0, "train_loss": None, "upload_bytes": 0, "upload_status": "pending"})
            global_sd = clone_state_dict(global_model.state_dict())
            download_bytes = state_dict_num_bytes(global_sd)  # bytes every client receives this round
            emit()

            local_states: list[dict[str, torch.Tensor]] = []
            local_counts: list[int] = []
            upload_sizes: list[int] = []
            client_epoch_losses: list[list[float]] = []
            state["server"]["status"] = "waiting_for_clients"
            for k, client in enumerate(clients):
                cs = client_states[k]
                cs.update({"status": "downloading", "download_bytes": download_bytes})
                state["workflow_step"] = "distribute"
                emit()
                local_model = self.build_model(cfg, c)
                local_model.load_state_dict(global_sd)  # copy of the current global parameters
                cs["status"] = "training"
                state["workflow_step"] = "local_training"
                emit()

                def progress(epoch: int, epochs: int, batch: int, batches: int, loss: float, _k: int = k, _r: int = r) -> None:
                    client_states[_k].update({"epoch": epoch, "batch": batch, "total_batches": batches, "train_loss": round(loss, 5)})
                    frac = (_k + ((epoch - 1) + batch / batches) / epochs) / num_clients
                    state["percent"] = round(100.0 * ((_r - 1) + frac) / cfg.rounds, 1)
                    emit(force=False)

                losses = train_on_pairs(
                    local_model, prep.data_norm, client.pairs, lookback=cfg.lookback, horizon=cfg.horizon,
                    target_channel=self.settings.flow_channel, epochs=cfg.local_epochs, learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size, seed=seed * 1000 + r * 10 + k, device=self.device,
                    progress_cb=progress, cancel_check=ctx.cancelled,
                )
                state["workflow_step"] = "local_update"
                local_sd = clone_state_dict(local_model.state_dict())
                up_bytes = state_dict_num_bytes(local_sd)
                cs.update({"status": "uploading", "upload_status": "uploading", "upload_bytes": up_bytes, "train_loss": round(losses[-1], 5)})
                state["workflow_step"] = "upload"
                emit()
                local_states.append(local_sd)
                local_counts.append(client.num_examples)
                upload_sizes.append(up_bytes)
                client_epoch_losses.append([round(l, 6) for l in losses])
                cs.update({"status": "uploaded", "upload_status": "uploaded"})
                emit()

            state["server"]["status"] = "aggregating"
            state["workflow_step"] = "aggregate"
            emit()
            new_global = fedavg_aggregate(local_states, local_counts)
            global_model.load_state_dict(new_global)

            state["server"]["status"] = "evaluating"
            state["workflow_step"] = "evaluate"
            emit()
            val = self._evaluate(global_model, cfg, prep, prep.val_pairs)
            comm = round_communication(r, download_bytes, upload_sizes, cumulative, param_count, raw_bytes)
            cumulative = comm["cumulative_bytes"]
            weights = [cnt / total_examples for cnt in local_counts]
            entry = {
                "round": r,
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_mse_normalized": val["mse_normalized"],
                "train_loss_mean": round(float(np.average([l[-1] for l in client_epoch_losses], weights=weights)), 6),
                "client_train_losses": [l[-1] for l in client_epoch_losses],
                "client_epoch_losses": client_epoch_losses,
                "client_num_examples": local_counts,
                "aggregation_weights": [round(w, 6) for w in weights],
                **comm,
                "duration_seconds": round(time.time() - round_start, 3),
                "timestamp": utc_now(),
            }
            history.append(entry)
            state["server"].update({"val_mae": val["mae"], "val_rmse": val["rmse"], "status": "round_complete", "cumulative_bytes": cumulative})
            state["workflow_step"] = "next_round" if r < cfg.rounds else "evaluate"
            for cs in client_states:
                cs["status"] = "completed"
            state["percent"] = round(100.0 * r / cfg.rounds, 1)
            emit()

        state["phase"] = "evaluating"
        state["message"] = "Evaluating the final global model on the chronological test period"
        emit()
        test = self._evaluate(global_model, cfg, prep, prep.test_pairs)
        best = min(history, key=lambda h: h["val_mae"]) if history else None
        model_path = self._save_model(global_model, ctx.experiment_id)
        state["phase"] = "completed"
        state["server"]["status"] = "completed"
        state["percent"] = 100
        emit()
        return {
            "model": model_info,
            "partitioning": {
                "strategy": clients[0].strategy if clients else strategy,
                "description": partition_description(strategy),
                "num_clients": num_clients,
                "total_examples": total_examples,
                "clients": [cl.summary() for cl in clients],
                "uses_test_data": False,
            },
            "history": history,
            "test_metrics": {"mae": test["mae"], "rmse": test["rmse"], "num_examples": test["num_examples"]},
            "best_validation": {"round": best["round"], "val_mae": best["val_mae"], "val_rmse": best["val_rmse"]} if best else None,
            "communication": summarize_federated(history, payload_bytes, num_clients),
            "predictions_sample": self._prediction_samples(global_model, cfg, prep),
            "model_path": model_path,
            "round_definition": f"One federated round = every client trains {cfg.local_epochs} local epoch(s), uploads its parameters, the server aggregates with FedAvg and evaluates on the validation windows.",
        }

    # ---------------------------------------------------------- predictions
    def load_model(self, record: dict[str, Any]) -> GRUForecaster:
        path = self.store.model_path(record["experiment_id"])
        if not path.is_file():
            raise FileNotFoundError("No saved model for this experiment (it may have failed or been cancelled).")
        cfg = ExperimentConfig(**record["config"])
        num_channels = int((record.get("dataset") or {}).get("shape", [0, 0, 3])[2])
        model = self.build_model(cfg, num_channels)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def predict_series(self, record: dict[str, Any], sensor_id: int, start: int | None, steps: int, horizon_step: int) -> dict[str, Any]:
        cfg = ExperimentConfig(**record["config"])
        _, data_norm, scaler, split = self.normalized()
        t_total, num_sensors, _ = data_norm.shape
        if sensor_id < 0 or sensor_id >= num_sensors:
            raise ValueError(f"sensor_id must be between 0 and {num_sensors - 1}.")
        lookback, horizon = cfg.lookback, cfg.horizon
        h = min(max(int(horizon_step), 1), horizon)
        test_start, test_end = split.test
        last_origin = test_end - lookback - horizon
        if last_origin < test_start:
            raise ValueError("Test period is too short for this lookback/horizon.")
        start = test_start if start is None else min(max(int(start), test_start), last_origin)
        steps = min(max(int(steps), 1), 2016)
        origins = np.arange(start, min(start + steps, last_origin + 1), dtype=np.int64)
        pairs = np.stack([origins, np.full_like(origins, sensor_id)], axis=1)
        model = self.load_model(record)
        _, actual, pred = predict_pairs(
            model, data_norm, pairs, lookback=lookback, horizon=horizon,
            target_channel=self.settings.flow_channel, scaler=scaler, device=self.device,
        )
        target_idx = origins + lookback + (h - 1)
        a = actual[:, h - 1]
        p = pred[:, h - 1]
        return {
            "experiment_id": record["experiment_id"],
            "method": record.get("method"),
            "sensor_id": sensor_id,
            "horizon_step": h,
            "horizon_minutes": h * self.dataset.sampling_interval_minutes,
            "test_period": {"start_index": test_start, "end_index": test_end, "first_origin": test_start, "last_origin": int(last_origin)},
            "start": int(start),
            "steps": int(origins.size),
            "indices": target_idx.tolist(),
            "timestamps": self.dataset.timestamps(target_idx, t_total),
            "actual": [round(float(v), 2) for v in a],
            "predicted": [round(float(v), 2) for v in p],
            "mae": round(mae_metric(a, p), 3),
            "rmse": round(rmse_metric(a, p), 3),
        }
