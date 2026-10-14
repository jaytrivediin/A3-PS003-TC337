"""Local JSON persistence and CSV export of experiment records."""
from __future__ import annotations

import csv
import io
import json
import threading
from pathlib import Path
from typing import Any

SUMMARY_CONFIG_KEYS = [
    "num_clients",
    "rounds",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "hidden_size",
    "num_layers",
    "lookback",
    "horizon",
    "max_train_examples",
    "max_eval_examples",
]


class ExperimentStore:
    def __init__(self, experiments_dir: Path, models_dir: Path) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.models_dir = Path(models_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def record_path(self, experiment_id: str) -> Path:
        return self.experiments_dir / f"{experiment_id}.json"

    def model_path(self, experiment_id: str) -> Path:
        return self.models_dir / f"{experiment_id}.pt"

    # ------------------------------------------------------------------- CRUD
    def save(self, record: dict[str, Any]) -> None:
        path = self.record_path(record["experiment_id"])
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, default=_json_default)
            tmp.replace(path)

    def load(self, experiment_id: str) -> dict[str, Any] | None:
        path = self.record_path(experiment_id)
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def delete(self, experiment_id: str) -> bool:
        removed = False
        for path in (self.record_path(experiment_id), self.model_path(experiment_id)):
            if path.is_file():
                path.unlink()
                removed = True
        return removed

    def list_records(self) -> list[dict[str, Any]]:
        records = []
        for path in self.experiments_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records

    def list_summaries(self) -> list[dict[str, Any]]:
        return [self.summary(r) for r in self.list_records()]

    # --------------------------------------------------------------- summary
    @staticmethod
    def summary(record: dict[str, Any]) -> dict[str, Any]:
        config = record.get("config", {})
        history = record.get("history") or []
        last = history[-1] if history else {}
        comm = record.get("communication") or {}
        return {
            "experiment_id": record.get("experiment_id"),
            "job_id": record.get("job_id"),
            "label": record.get("label"),
            "status": record.get("status"),
            "method": record.get("method"),
            "method_label": record.get("method_label"),
            "seed": record.get("seed"),
            "seed_mode": record.get("seed_mode"),
            "created_at": record.get("created_at"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "duration_seconds": record.get("duration_seconds"),
            "config": {k: config.get(k) for k in SUMMARY_CONFIG_KEYS},
            "partition_strategy": (record.get("partitioning") or {}).get("strategy"),
            "train_examples": (record.get("data") or {}).get("train_examples"),
            "train_sampled": (record.get("data") or {}).get("train_sampled"),
            "test_examples": (record.get("data") or {}).get("test_examples"),
            "rounds_completed": len(history),
            "final_val_mae": last.get("val_mae"),
            "final_val_rmse": last.get("val_rmse"),
            "test_mae": (record.get("test_metrics") or {}).get("mae"),
            "test_rmse": (record.get("test_metrics") or {}).get("rmse"),
            "communication": {
                "type": comm.get("type"),
                "total_bytes": comm.get("total_bytes"),
                "total_download_bytes": comm.get("total_download_bytes"),
                "total_upload_bytes": comm.get("total_upload_bytes"),
                "estimated_raw_data_upload_bytes": comm.get("estimated_raw_data_upload_bytes"),
                "model_payload_bytes_per_transfer": comm.get("model_payload_bytes_per_transfer"),
            },
            "model_parameter_count": (record.get("model") or {}).get("parameter_count"),
            "error": record.get("error"),
        }

    # ------------------------------------------------------------------- CSV
    @staticmethod
    def record_to_csv(record: dict[str, Any]) -> str:
        """Per-round CSV for a single experiment with summary columns repeated."""
        summary = ExperimentStore.summary(record)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        header = [
            "experiment_id", "method", "seed", "status", "num_clients", "rounds", "local_epochs",
            "learning_rate", "batch_size", "hidden_size", "train_examples", "test_examples",
            "test_mae", "test_rmse", "total_communication_bytes", "round", "val_mae", "val_rmse",
            "train_loss", "round_download_bytes", "round_upload_bytes", "round_bytes", "cumulative_bytes",
        ]
        writer.writerow(header)
        history = record.get("history") or []
        base = [
            summary["experiment_id"], summary["method"], summary["seed"], summary["status"],
            summary["config"].get("num_clients"), summary["config"].get("rounds"), summary["config"].get("local_epochs"),
            summary["config"].get("learning_rate"), summary["config"].get("batch_size"), summary["config"].get("hidden_size"),
            summary["train_examples"], summary["test_examples"], summary["test_mae"], summary["test_rmse"],
            summary["communication"].get("total_bytes"),
        ]
        if not history:
            writer.writerow(base + ["", "", "", "", "", "", "", ""])
        for entry in history:
            writer.writerow(
                base
                + [
                    entry.get("round"), entry.get("val_mae"), entry.get("val_rmse"), entry.get("train_loss_mean"),
                    entry.get("total_download_bytes", 0), entry.get("total_upload_bytes", 0),
                    entry.get("round_bytes", 0), entry.get("cumulative_bytes", 0),
                ]
            )
        return buffer.getvalue()

    def all_summaries_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "experiment_id", "job_id", "label", "status", "method", "seed", "seed_mode", "created_at",
                "duration_seconds", "num_clients", "rounds", "local_epochs", "learning_rate", "batch_size",
                "hidden_size", "partition_strategy", "train_examples", "test_examples", "final_val_mae",
                "final_val_rmse", "test_mae", "test_rmse", "communication_type", "total_bytes",
                "total_download_bytes", "total_upload_bytes", "estimated_raw_data_upload_bytes",
            ]
        )
        for s in self.list_summaries():
            cfg = s["config"]
            comm = s["communication"]
            writer.writerow(
                [
                    s["experiment_id"], s["job_id"], s["label"], s["status"], s["method"], s["seed"], s["seed_mode"],
                    s["created_at"], s["duration_seconds"], cfg.get("num_clients"), cfg.get("rounds"),
                    cfg.get("local_epochs"), cfg.get("learning_rate"), cfg.get("batch_size"), cfg.get("hidden_size"),
                    s["partition_strategy"], s["train_examples"], s["test_examples"], s["final_val_mae"],
                    s["final_val_rmse"], s["test_mae"], s["test_rmse"], comm.get("type"), comm.get("total_bytes"),
                    comm.get("total_download_bytes"), comm.get("total_upload_bytes"),
                    comm.get("estimated_raw_data_upload_bytes"),
                ]
            )
        return buffer.getvalue()


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
