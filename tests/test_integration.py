from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_real_world_usage():
    texts = [
        "I love this!",
        "This is awful.",
        "Not bad, could be better.",
        "Absolutely fantastic experience.",
        "Terrible. Would not recommend."
    ]

    for t in texts:
        resp = client.post("/predict", json={"text": t})
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment_score" in data
        assert "label" in data
