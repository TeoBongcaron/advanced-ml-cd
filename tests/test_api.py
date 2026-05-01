from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_positive():
    payload = {"text": "I really love this product, it is amazing!"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment_score" in data
    assert "label" in data

def test_predict_negative():
    payload = {"text": "This is terrible and I hate it."}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment_score" in data
    assert "label" in data
