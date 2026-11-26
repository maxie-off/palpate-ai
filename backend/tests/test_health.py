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


def test_agent_route_delegates_and_validates() -> None:
    from app.agent.types import AgentResult

    def fake_run(body: object) -> AgentResult:
        return AgentResult(task="qa", status="insufficient_evidence", summary="no sources")

    import app.api.routes as routes

    original = routes.run_agent
    routes.run_agent = fake_run  # type: ignore[assignment]
    try:
        client = TestClient(app)
        ok = client.post("/agent/run", json={"task": "qa", "message": "What are milia?"})
        rejected = client.post("/agent/run", json={"task": "not-a-task"})
    finally:
        routes.run_agent = original

    assert ok.status_code == 200
    assert ok.json()["status"] == "insufficient_evidence"
    assert rejected.status_code == 422
