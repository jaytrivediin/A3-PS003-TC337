"""A minimal background job manager for a local prototype (single worker thread).

One job = one or more experiment runs (methods x seeds). Jobs execute sequentially
so that CPU resources are not oversubscribed and duplicate runs cannot be started
accidentally. Live progress is kept in memory and merged into API responses.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.experiments.config import METHOD_LABELS, ExperimentConfig


class JobAlreadyRunningError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


@dataclass
class RunPlan:
    experiment_id: str
    method: str
    seed: int
    status: str = "queued"
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "method_label": METHOD_LABELS.get(self.method, self.method),
            "seed": self.seed,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class Job:
    job_id: str
    config: ExperimentConfig
    runs: list[RunPlan]
    created_at: str = field(default_factory=utc_now)
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    current_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        completed = sum(1 for r in self.runs if r.status in ("completed", "failed", "cancelled"))
        return {
            "job_id": self.job_id,
            "status": self.status,
            "label": self.config.label,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "config": self.config.dump(),
            "seeds": self.config.seeds(),
            "methods": self.config.methods(),
            "runs": [r.to_dict() for r in self.runs],
            "current_index": self.current_index,
            "runs_total": len(self.runs),
            "runs_finished": completed,
            "percent": round(100.0 * completed / max(len(self.runs), 1), 1),
            "cancel_requested": self.cancel_event.is_set(),
        }


class RunContext:
    """Handed to the runner: identifiers, config, progress reporting and cancellation."""

    def __init__(self, manager: "JobManager", job: Job, run: RunPlan) -> None:
        self.manager = manager
        self.job = job
        self.run = run
        self.experiment_id = run.experiment_id
        self.job_id = job.job_id
        self.method = run.method
        self.seed = run.seed
        self.config = job.config
        self._last_report = 0.0

    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def report(self, progress: dict[str, Any], force: bool = True, min_interval: float = 0.15) -> None:
        now = time.time()
        if not force and now - self._last_report < min_interval:
            return
        self._last_report = now
        with self.manager.lock:
            self.run.progress = progress


class JobManager:
    def __init__(self, runner: Any, store: Any) -> None:
        self.runner = runner
        self.store = store
        self.lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name="fedtraffic-jobs", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._execute(job)
            finally:
                self._queue.task_done()

    def _execute(self, job: Job) -> None:
        with self.lock:
            job.status = "running"
            job.started_at = utc_now()
        for index, run in enumerate(job.runs):
            with self.lock:
                job.current_index = index
                if job.cancel_event.is_set():
                    run.status = "cancelled"
                    self._persist_status(run, "cancelled", "Cancelled before start.")
                    continue
                run.status = "running"
                run.started_at = utc_now()
            ctx = RunContext(self, job, run)
            try:
                record = self.runner.run(ctx)
                status = record.get("status", "failed")
                error = record.get("error")
            except Exception as exc:  # noqa: BLE001 - never kill the worker thread
                status, error = "failed", f"{type(exc).__name__}: {exc}"
                self._persist_status(run, status, error)
            with self.lock:
                run.status = status
                run.error = error
                run.finished_at = utc_now()
                run.progress = {**run.progress, "phase": status, "percent": 100 if status == "completed" else run.progress.get("percent", 0)}
        with self.lock:
            statuses = {r.status for r in job.runs}
            if job.cancel_event.is_set() and "cancelled" in statuses:
                job.status = "cancelled"
            elif statuses == {"completed"}:
                job.status = "completed"
            elif "completed" in statuses:
                job.status = "completed_with_errors"
            else:
                job.status = "failed"
            job.finished_at = utc_now()

    def _persist_status(self, run: RunPlan, status: str, error: str | None) -> None:
        record = self.store.load(run.experiment_id)
        if record is not None:
            record["status"] = status
            record["error"] = error
            record["finished_at"] = utc_now()
            self.store.save(record)

    # -------------------------------------------------------------- queries
    def submit(self, config: ExperimentConfig) -> Job:
        with self.lock:
            active = self.active_job()
            if active is not None:
                raise JobAlreadyRunningError(
                    f"Job {active.job_id} is still {active.status}. Wait for it to finish or cancel it first."
                )
            job_id = new_id("job")
            runs = []
            for item in config.plan():
                experiment_id = new_id("exp")
                runs.append(RunPlan(experiment_id=experiment_id, method=item["method"], seed=item["seed"]))
                self.store.save(
                    {
                        "experiment_id": experiment_id,
                        "job_id": job_id,
                        "label": config.label,
                        "status": "queued",
                        "method": item["method"],
                        "method_label": METHOD_LABELS.get(item["method"], item["method"]),
                        "seed": item["seed"],
                        "seed_mode": config.seed_mode,
                        "created_at": utc_now(),
                        "config": config.dump(),
                        "history": [],
                        "error": None,
                    }
                )
            job = Job(job_id=job_id, config=config, runs=runs)
            self._jobs[job_id] = job
            self._ensure_worker()
            self._queue.put(job)
            return job

    def active_job(self) -> Job | None:
        with self.lock:
            for job in self._jobs.values():
                if job.status in ("queued", "running"):
                    return job
        return None

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        with self.lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in ("queued", "running"):
                job.cancel_event.set()
            return job

    def live_progress(self, experiment_id: str) -> dict[str, Any] | None:
        with self.lock:
            for job in self._jobs.values():
                for run in job.runs:
                    if run.experiment_id == experiment_id:
                        return {"status": run.status, "job_id": job.job_id, **run.progress}
        return None

    def run_status(self, experiment_id: str) -> str | None:
        with self.lock:
            for job in self._jobs.values():
                for run in job.runs:
                    if run.experiment_id == experiment_id:
                        return run.status
        return None
