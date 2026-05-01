from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from .model import SentimentModel

app = FastAPI(title="Sentiment Analysis API")

# Load model at startup
model = SentimentModel("models/sentiment.onnx")

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment_score: float
    label: str

def dummy_text_to_features(text: str) -> np.ndarray:
    # TODO: replace with real preprocessing
    # simple toy feature: length of text
    length = len(text)
    return np.array([[length]], dtype=np.float32)

@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(payload: TextInput):
    features = dummy_text_to_features(payload.text)
    score = model.predict(features)

    label = "positive" if score >= 0.5 else "negative"

    return PredictionOutput(
        sentiment_score=score,
        label=label
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

