from fastapi import FastAPI
from pydantic import BaseModel
from .model import SentimentModel

app = FastAPI(title="Sentiment Analysis API")

# Load ONNX model at startup
model = SentimentModel("models/sentiment.onnx")

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment_score: float
    label: str

@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(payload: TextInput):
    score = model.predict(payload.text)
    label = "positive" if score >= 0.5 else "negative"

    return PredictionOutput(
        sentiment_score=score,
        label=label
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}


