"""FedTraffic FastAPI application factory.

Run locally with:  uvicorn app.main:app --reload --port 8000   (from the backend directory)
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_dataset import router as dataset_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_health import router as health_router
from app.core.config import Settings, get_settings
from app.data.pems04 import MANUAL_INSTRUCTIONS, DatasetNotAvailableError, DatasetValidationError, PEMS04Dataset
from app.experiments.jobs import JobManager
from app.experiments.runner import ExperimentRunner
from app.experiments.store import ExperimentStore

API_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="FedTraffic API",
        description="Federated traffic-flow forecasting on PEMS04 with communication-cost evaluation.",
        version=API_VERSION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dataset = PEMS04Dataset(
        settings.dataset_path,
        settings.pems04_url,
        start_date=settings.pems04_start_date,
        expected_shape=(settings.pems04_expected_timesteps, settings.pems04_expected_sensors, settings.pems04_expected_features),
        sampling_interval_minutes=settings.sampling_interval_minutes,
        flow_channel=settings.flow_channel,
        split_ratios=(settings.train_ratio, settings.val_ratio, settings.test_ratio),
    )
    store = ExperimentStore(settings.experiments_dir, settings.models_dir)
    runner = ExperimentRunner(dataset, settings, store)
    jobs = JobManager(runner, store)

    app.state.settings = settings
    app.state.dataset = dataset
    app.state.store = store
    app.state.runner = runner
    app.state.jobs = jobs

    app.include_router(health_router, prefix="/api")
    app.include_router(dataset_router, prefix="/api")
    app.include_router(experiments_router, prefix="/api")

    @app.exception_handler(DatasetNotAvailableError)
    async def _not_available(_: Request, exc: DatasetNotAvailableError):
        return JSONResponse(
            status_code=409,
            content={"detail": {"code": "dataset_not_available", "message": str(exc), "instructions": MANUAL_INSTRUCTIONS}},
        )

    @app.exception_handler(DatasetValidationError)
    async def _invalid(_: Request, exc: DatasetValidationError):
        return JSONResponse(status_code=422, content={"detail": {"code": "dataset_invalid", "message": str(exc)}})

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "fedtraffic-backend", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
