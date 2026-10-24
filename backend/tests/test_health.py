from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unimplemented_routes_return_501() -> None:
    client = TestClient(app)
    assert client.post("/documents").status_code == 501
    assert client.post("/worksheets").status_code == 501
