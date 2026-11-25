from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ml_pipeline import BusPassengerPipeline

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

pipeline = BusPassengerPipeline()


def ensure_pipeline_ready() -> None:
    if not pipeline.metrics:
        pipeline.fit()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_pipeline_ready()
    yield


app = FastAPI(
    title="Bus Passenger Count Prediction API",
    description="Interactive ML analytics and live passenger crowding prediction API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_ready_middleware(request: Request, call_next):
    ensure_pipeline_ready()
    return await call_next(request)


class PredictRequest(BaseModel):
    stop_id: str | None = None
    custom_historical_mean: float | None = Field(default=None, ge=0, le=200)
    bus_capacity: int = Field(default=50, ge=10, le=150)
    day: str = "Monday"
    weather: str = "Clear"
    selected_model: str | None = None


@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/api/overview")
def get_overview() -> dict:
    return pipeline.overview


@app.get("/api/models")
def get_models(model: str | None = Query(default=None)) -> dict:
    selected = model if model in pipeline.model_diagnostics else pipeline.best_model_name
    return {
        "metrics": pipeline.metrics,
        "selected_model": selected,
        "diagnostics": pipeline.model_diagnostics[selected],
    }


@app.get("/api/stops")
def get_stops() -> dict:
    return {
        "total": len(pipeline.stops_catalog),
        "stops": pipeline.stops_catalog,
    }


@app.get("/api/analytics")
def get_analytics() -> dict:
    return pipeline.analytics


@app.get("/api/predictions")
def get_predictions(
    model: str | None = Query(default=None),
    search: str = Query(default=""),
    weather: str = Query(default=""),
    stop_id: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    return pipeline.get_predictions_table(
        model_name=model,
        search=search,
        weather=weather,
        stop_id=stop_id,
        limit=limit,
        offset=offset,
    )


@app.post("/api/predict")
def post_predict(req: PredictRequest) -> dict:
    return pipeline.predict_live(
        stop_id=req.stop_id,
        custom_historical_mean=req.custom_historical_mean,
        bus_capacity=req.bus_capacity,
        day=req.day,
        weather=req.weather,
        selected_model=req.selected_model,
    )


@app.get("/api/product/commuter")
def get_commuter_forecast(
    stop_id: str | None = Query(default=None),
    bus_capacity: int | None = Query(default=None),
    day: str = Query(default="Monday"),
    weather: str = Query(default="Clear"),
    departure_time: str = Query(default="07:30"),
) -> dict:
    try:
        return pipeline.get_commuter_forecast(
            stop_id=stop_id,
            bus_capacity=bus_capacity,
            day=day,
            weather=weather,
            departure_time=departure_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/product/operator")
def get_operator_dispatch(
    weather: str = Query(default="Clear"),
    day: str = Query(default="Monday"),
) -> dict:
    return pipeline.get_operator_dispatch(weather=weather, day=day)


@app.post("/api/retrain")
async def post_retrain(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        pipeline.fit(file_bytes=content, filename=file.filename)
        return {
            "status": "retrained",
            "overview": pipeline.overview,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run("app:app", host=host, port=port, reload=False)
