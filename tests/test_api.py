from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Congratulations, claim your free prize now!"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["label"] in {"ham", "spam"}
    assert 0.0 <= body["spam_probability"] <= 1.0


def test_predict_rejects_blank_text() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
