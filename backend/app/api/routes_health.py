"""Health endpoint."""
from __future__ import annotations

from datetime import datetime, timezone

import torch
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    dataset = request.app.state.dataset
    jobs = request.app.state.jobs
    active = jobs.active_job()
    return {
        "status": "ok",
        "service": "fedtraffic-backend",
        "version": request.app.version,
        "time": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(request.app.state.runner.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "dataset_file_exists": dataset.file_exists(),
        "dataset_loaded": dataset.is_loaded(),
        "active_job_id": active.job_id if active else None,
    }
