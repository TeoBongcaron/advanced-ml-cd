from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_sql_injection():
    payload = {"text": "'; DROP TABLE users; --"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

def test_script_injection():
    payload = {"text": "<script>alert('x')</script>"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

def test_json_injection():
    payload = {"text": '{"key": "value"}'}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

def test_binary_input():
    payload = {"text": "\x00\x01\x02\x03"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

def test_extremely_large_payload():
    payload = {"text": "x" * 500_000}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
