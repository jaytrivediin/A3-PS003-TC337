"""Experiment / job endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.data.pems04 import DatasetError
from app.experiments.config import ExperimentConfig, presets
from app.experiments.jobs import JobAlreadyRunningError
from app.experiments.runner import WORKFLOW_STEPS

router = APIRouter(tags=["experiments"])

FINAL_STATES = {"completed", "failed", "cancelled"}


@router.get("/config/defaults")
def config_defaults(request: Request):
    settings = request.app.state.settings
    runner = request.app.state.runner
    return {
        **presets(),
        "split": {"train": settings.train_ratio, "val": settings.val_ratio, "test": settings.test_ratio},
        "eval_sample_seed": settings.eval_sample_seed,
        "device": str(runner.device),
        "workflow_steps": WORKFLOW_STEPS,
    }


@router.post("/experiments", status_code=202)
def create_experiment(request: Request, config: ExperimentConfig):
    dataset = request.app.state.dataset
    try:
        dataset.load()
    except DatasetError as exc:
        raise HTTPException(status_code=409, detail={"code": "dataset_not_available", "message": str(exc)}) from exc
    jobs = request.app.state.jobs
    try:
        job = jobs.submit(config)
    except JobAlreadyRunningError as exc:
        active = jobs.active_job()
        raise HTTPException(
            status_code=409,
            detail={"code": "job_running", "message": str(exc), "job_id": active.job_id if active else None},
        ) from exc
    return {"job": job.to_dict(), "experiment_ids": [r.experiment_id for r in job.runs]}


@router.get("/experiments")
def list_experiments(request: Request, limit: int = Query(300, ge=1, le=2000)):
    store = request.app.state.store
    jobs = request.app.state.jobs
    summaries = store.list_summaries()
    for s in summaries:
        live = jobs.run_status(s["experiment_id"])
        if live is not None:
            s["status"] = live
    return {"experiments": summaries[:limit], "count": len(summaries)}


def _load_record(request: Request, experiment_id: str) -> dict:
    record = request.app.state.store.load(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    live = request.app.state.jobs.live_progress(experiment_id)
    if live is not None:
        record["progress"] = live
        if record.get("status") not in FINAL_STATES:
            record["status"] = live.get("status", record.get("status"))
    return record


@router.get("/experiments/{experiment_id}")
def get_experiment(request: Request, experiment_id: str):
    return _load_record(request, experiment_id)


@router.get("/experiments/{experiment_id}/results")
def get_results(request: Request, experiment_id: str):
    record = _load_record(request, experiment_id)
    keys = [
        "experiment_id", "job_id", "label", "status", "method", "method_label", "seed", "seed_mode", "config",
        "dataset", "data", "model", "partitioning", "history", "test_metrics", "best_validation",
        "communication", "predictions_sample", "round_definition", "duration_seconds", "error",
    ]
    return {k: record.get(k) for k in keys}


@router.get("/experiments/{experiment_id}/export")
def export_experiment(request: Request, experiment_id: str):
    record = _load_record(request, experiment_id)
    csv_text = request.app.state.store.record_to_csv(record)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.csv"'},
    )


@router.get("/experiments/{experiment_id}/predictions")
def experiment_predictions(
    request: Request,
    experiment_id: str,
    sensor_id: int = Query(0, ge=0),
    start: int | None = Query(None, ge=0),
    steps: int = Query(288, ge=1, le=2016),
    horizon_step: int = Query(1, ge=1),
):
    record = _load_record(request, experiment_id)
    if record.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Predictions are only available for completed experiments.")
    try:
        return request.app.state.runner.predict_series(record, sensor_id, start, steps, horizon_step)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/experiments/{experiment_id}")
def delete_experiment(request: Request, experiment_id: str):
    jobs = request.app.state.jobs
    if jobs.run_status(experiment_id) in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Cannot delete an experiment that is queued or running.")
    if not request.app.state.store.delete(experiment_id):
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return {"deleted": experiment_id}


@router.get("/export/experiments.csv")
def export_all(request: Request):
    csv_text = request.app.state.store.all_summaries_csv()
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fedtraffic_experiments.csv"'},
    )


@router.get("/jobs")
def list_jobs(request: Request):
    return {"jobs": [j.to_dict() for j in request.app.state.jobs.list()]}


@router.get("/jobs/active")
def active_job(request: Request):
    job = request.app.state.jobs.active_job()
    return {"job": job.to_dict() if job else None}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found (jobs live in memory until restart).")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str):
    job = request.app.state.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()
