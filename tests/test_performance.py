import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_performance_under_load():
    start = time.time()

    for _ in range(100):
        resp = client.post("/predict", json={"text": "This is a test"})
        assert resp.status_code == 200

    duration = time.time() - start

    # Expect 100 requests to finish under 5 seconds
    assert duration < 5
