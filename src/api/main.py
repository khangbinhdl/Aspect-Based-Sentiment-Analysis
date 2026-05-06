from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.config import settings
from src.models.model_loader import ABSAService


app = FastAPI(
    title="ABSA Inference API",
    description="Inference API for ABTE + ABSC (Aspect-Based Sentiment Analysis).",
    version="2.0.0",
)

model_service = ABSAService()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input sentence")
    abte_model_name: str = Field(default=settings.default_abte_model_name)
    absc_model_name: str = Field(default=settings.default_absc_model_name)
    term: Optional[str] = Field(default=None, description="Optional aspect term override")
    device: Optional[str] = Field(default=None, description="cpu, cuda, mps, or auto")


class SentimentResult(BaseModel):
    term: str
    sentiment_id: int
    sentiment: str
    sentiment_score: float


class PredictResponse(BaseModel):
    sentence: str
    abte_model: str
    absc_model: str
    tokens: List[str]
    labels: List[str]
    terms: List[str]
    results: List[SentimentResult]
    message: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return model_service.list_models()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        output = model_service.predict_absa(
            sentence=payload.text,
            abte_model_name=payload.abte_model_name,
            absc_model_name=payload.absc_model_name,
            term=payload.term,
                device=payload.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**output)
