from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    assert "sentiment_score" in resp.json()

def test_whitespace_only():
    resp = client.post("/predict", json={"text": "     "})
    assert resp.status_code == 200

def test_very_long_text():
    text = "good " * 10000
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200

def test_non_english_text():
    resp = client.post("/predict", json={"text": "これはとても良いです"})
    assert resp.status_code == 200

def test_numeric_input():
    resp = client.post("/predict", json={"text": "1234567890"})
    assert resp.status_code == 200
