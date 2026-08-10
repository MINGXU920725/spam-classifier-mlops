"""HTTP API for the spam classifier."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.predict import SpamPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "spam_classifier.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    app.state.predictor = SpamPredictor(model_path)
    yield


app = FastAPI(
    title="Spam Classifier API",
    description="Classify an SMS message as ham or spam.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)

    @field_validator("text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictionResponse(BaseModel):
    label: str
    spam_probability: float
    model_version: str


@app.get("/", tags=["system"])
def root() -> dict:
    return {"service": "spam-classifier", "docs": "/docs"}


@app.get("/health", tags=["system"])
def health(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {
        "status": "healthy",
        "model_version": predictor.model_version,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionRequest, request: Request) -> dict:
    try:
        return request.app.state.predictor.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
