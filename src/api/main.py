from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.config import settings
from src.models.model_loader import ABTEModelService


app = FastAPI(
    title="ABTE Inference API",
    description="Inference API for Aspect-Based Term Extraction (LSTM/DistilBERT).",
    version="1.0.0",
)

model_service = ABTEModelService()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input sentence")
    model_name: str = Field(default=settings.default_model_name)


class PredictResponse(BaseModel):
    model_name: str
    tokens: List[str]
    labels: List[str]
    terms: List[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {"models": model_service.list_models()}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        output = model_service.predict(payload.text, payload.model_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**output)
