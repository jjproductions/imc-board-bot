from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock, patch


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


def test_ingest_async_executor():
    with patch("app.routes.api.qdrant") as mock_qdrant:
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant.get_collections.return_value = mock_collections
        
        client = TestClient(app)
        
        payload = {
            "source_id": "test-doc-123",
            "title": "Test Document",
            "blocks": [
                {
                    "id": "b1",
                    "type": "heading",
                    "text": "Introduction",
                    "level": 1
                },
                {
                    "id": "b2",
                    "type": "paragraph",
                    "text": "This is a paragraph under introduction heading."
                }
            ]
        }
        
        r = client.post("/api/ingest", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["chunks_inserted"] == 1
        assert data["source_id"] == "test-doc-123"
        
        assert mock_qdrant.upsert.called


def test_delete_collection():
    with patch("app.routes.api.qdrant") as mock_qdrant:
        client = TestClient(app)
        r = client.delete("/api/collections/test-collection")
        assert r.status_code == 204
        mock_qdrant.delete_collection.assert_called_once_with("test-collection")
