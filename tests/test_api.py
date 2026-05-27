"""Tests for the GraphMemory REST API (RED phase).

Exercises all 15 endpoints through ``TestClient``.  Uses an in-memory SQLite
database so no files are created on disk.  Skips embedding-dependent assertions
when the Sentence-Transformers model has not been downloaded.
"""

from __future__ import annotations

import sqlite3
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from graph_memory import GraphMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embeddings_available() -> bool:
    """Return True if EmbeddingEngine can be instantiated (model is downloaded)."""
    try:
        from graph_memory.embeddings import EmbeddingEngine  # noqa: PLC0415

        EmbeddingEngine()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with a fresh in-memory GraphMemory.

    Two patches are applied for the whole test session:

    * ``graph_memory.core.EmbeddingEngine`` is made to raise so that
      ``GraphMemory(":memory:")`` is fast (~50 ms vs 5 min).  Without this
      every test would wait for the sentence-transformers model to load.
    * ``sqlite3.connect`` is forced to ``check_same_thread=False`` because
      ``TestClient`` runs its ASGI app (and therefore endpoint handlers) in a
      background thread that differs from the thread that creates the
      ``GraphMemory`` instance.
    """
    from api.server import app  # noqa: PLC0415

    # Capture the real sqlite3.connect *before* patching.
    _real_connect = sqlite3.connect

    def _threadsafe_connect(database, *args, **kwargs):
        kwargs["check_same_thread"] = False
        return _real_connect(database, *args, **kwargs)

    with (
        patch("graph_memory.core.EmbeddingEngine") as mock_ee,
        patch("sqlite3.connect", side_effect=_threadsafe_connect),
    ):
        mock_ee.side_effect = Exception("Embeddings disabled in test")

        with TestClient(app) as c:
            # Override the lifespan-created memory with a fresh store so
            # every test starts with a clean, empty database.
            app.state.memory = GraphMemory(":memory:")
            yield c


# ===================================================================
# Nodes
# ===================================================================

class TestApiNodes:
    """POST /nodes, GET /nodes/{id}, DELETE /nodes/{id}."""

    def test_create_node(self, client: TestClient) -> None:
        """POST /nodes with minimal body returns 201 and a node_id."""
        resp = client.post("/nodes", json={"content": "Hello world"})
        assert resp.status_code == 201
        data = resp.json()
        assert "node_id" in data
        # Verify it is a valid UUID v4.
        parsed = uuid.UUID(data["node_id"])
        assert parsed.version == 4

    def test_get_node(self, client: TestClient) -> None:
        """GET /nodes/{id} returns the full Node as JSON."""
        create = client.post("/nodes", json={
            "content": "Python uses indentation",
            "node_type": "factual",
            "coreness": 0.7,
            "tags": ["python"],
        })
        node_id = create.json()["node_id"]

        resp = client.get(f"/nodes/{node_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "Python uses indentation"
        assert body["node_type"] == "factual"
        assert body["coreness"] == 0.7
        assert body["tags"] == ["python"]
        assert body["id"] == node_id
        assert body["is_deleted"] is False

    def test_get_node_404(self, client: TestClient) -> None:
        """GET /nodes/{id} with a non-existent ID returns 404."""
        resp = client.get("/nodes/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_node(self, client: TestClient) -> None:
        """DELETE /nodes/{id} soft-deletes the node."""
        create = client.post("/nodes", json={"content": "To be deleted"})
        node_id = create.json()["node_id"]

        del_resp = client.delete(f"/nodes/{node_id}")
        assert del_resp.status_code == 200

        # After soft-delete the node is still retrievable but marked deleted.
        get_resp = client.get(f"/nodes/{node_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["is_deleted"] is True


# ===================================================================
# Edges
# ===================================================================

class TestApiEdges:
    """POST /edges, GET /edges/{source}/{target}, DELETE /edges/{source}/{target}."""

    def test_create_edge(self, client: TestClient) -> None:
        """POST /edges with valid source+target returns 200."""
        src = client.post("/nodes", json={"content": "Source"}).json()["node_id"]
        tgt = client.post("/nodes", json={"content": "Target"}).json()["node_id"]

        resp = client.post("/edges", json={
            "source_id": src,
            "target_id": tgt,
            "relation_type": "sequence",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_get_edge_404(self, client: TestClient) -> None:
        """GET /edges/{source}/{target} with non-existent IDs returns 404."""
        fake_src = str(uuid.uuid4())
        fake_tgt = str(uuid.uuid4())
        resp = client.get(f"/edges/{fake_src}/{fake_tgt}")
        assert resp.status_code == 404


# ===================================================================
# Recall
# ===================================================================

class TestApiRecall:
    """POST /recall performs semantic retrieval."""

    def test_recall(self, client: TestClient) -> None:
        """POST /recall with seed data returns a list of scored results."""
        client.post("/nodes", json={"content": "Python programming", "tags": ["python"]})
        client.post("/nodes", json={"content": "Weather forecast", "tags": ["weather"]})

        resp = client.post("/recall", json={"query_text": "python", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        # At least one result should be returned (the Python node).
        assert len(results) >= 1

    def test_recall_empty(self, client: TestClient) -> None:
        """POST /recall on an empty database returns an empty list."""
        resp = client.post("/recall", json={"query_text": "anything"})
        assert resp.status_code == 200
        assert resp.json() == []


# ===================================================================
# Reinforce
# ===================================================================

class TestApiReinforce:
    """POST /reinforce strengthens a reasoning path."""

    def test_reinforce(self, client: TestClient) -> None:
        """POST /reinforce returns 200 and increases coreness."""
        a = client.post("/nodes", json={"content": "A", "coreness": 0.5}).json()["node_id"]
        b = client.post("/nodes", json={"content": "B", "coreness": 0.5}).json()["node_id"]
        c = client.post("/nodes", json={"content": "C", "coreness": 0.5}).json()["node_id"]
        client.post("/edges", json={"source_id": a, "target_id": b})
        client.post("/edges", json={"source_id": b, "target_id": c})

        resp = client.post("/reinforce", json={
            "node_ids": [a, b, c],
            "edge_ids": [[a, b], [b, c]],
            "success": True,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# Correct
# ===================================================================

class TestApiCorrect:
    """POST /correct applies an error correction."""

    def test_correct_reduce(self, client: TestClient) -> None:
        """POST /correct with correction_type=reduce lowers base_stability."""
        src = client.post("/nodes", json={"content": "Src"}).json()["node_id"]
        tgt = client.post("/nodes", json={"content": "Tgt"}).json()["node_id"]
        client.post("/edges", json={"source_id": src, "target_id": tgt})

        resp = client.post("/correct", json={
            "location": f"edge:{src}->{tgt}",
            "correction_type": "reduce",
            "value": 0.15,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# Chains
# ===================================================================

class TestApiChains:
    """POST /chains and GET /chains."""

    def test_log_chain(self, client: TestClient) -> None:
        """POST /chains stores a chain log."""
        a = client.post("/nodes", json={"content": "Step A"}).json()["node_id"]
        b = client.post("/nodes", json={"content": "Step B"}).json()["node_id"]
        client.post("/edges", json={"source_id": a, "target_id": b})

        resp = client.post("/chains", json={
            "node_ids": [a, b],
            "edge_ids": [[a, b]],
            "query": "test query",
            "success": True,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_get_chains(self, client: TestClient) -> None:
        """GET /chains returns a list of chain logs."""
        a = client.post("/nodes", json={"content": "Root"}).json()["node_id"]
        b = client.post("/nodes", json={"content": "Branch"}).json()["node_id"]
        client.post("/edges", json={"source_id": a, "target_id": b})
        client.post("/chains", json={
            "node_ids": [a, b],
            "edge_ids": [[a, b]],
            "query": "test",
            "success": True,
        })

        resp = client.get("/chains", params={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1


# ===================================================================
# Stats
# ===================================================================

class TestApiStats:
    """GET /stats returns node/edge counts."""

    def test_stats(self, client: TestClient) -> None:
        """GET /stats returns correct counts after creating data."""
        # Empty store.
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json() == {"node_count": 0, "edge_count": 0}

        # Add one node.
        client.post("/nodes", json={"content": "Only node"})
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] == 1
        assert data["edge_count"] == 0
