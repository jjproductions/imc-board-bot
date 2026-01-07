from fastapi.testclient import TestClient
from app.main import app


def test_crud():
    client = TestClient(app)

    # create
    r = client.post("/api/chunks", json={"name": "Test", "description": "d"})
    assert r.status_code == 201
    data = r.json()
    assert data["id"] == 1

    # list
    r = client.get("/api/chunks")
    assert r.status_code == 200
    assert len(r.json()) == 1

    # get
    r = client.get("/api/chunks/1")
    assert r.status_code == 200

    # update
    r = client.put("/api/chunks/1", json={"name": "Updated", "description": "x"})
    assert r.status_code == 200
    assert r.json()["name"] == "Updated"

    # delete
    r = client.delete("/api/chunks/1")
    assert r.status_code == 204
    r = client.get("/api/chunks/1")
    assert r.status_code == 404
