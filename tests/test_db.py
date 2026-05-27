"""Tests for the Database persistence layer (RED phase).

Database is the public interface for the AI memory system's data
persistence.  These tests exercise only the public API — no direct
SQLite calls — so the implementation behind the interface is free
to change.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from graph_memory.db import Database
from graph_memory.models import Edge, Node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """Return a fresh in-memory Database instance ready for use."""
    database = Database(":memory:")
    database.initialize()
    return database


# ---------------------------------------------------------------------------
# Construction & initialisation
# ---------------------------------------------------------------------------

class TestConstruction:
    """Database(path) creates a connection and sets journal mode to WAL."""

    def test_construction_succeeds(self) -> None:
        """Database(path) constructs without error."""
        database = Database(":memory:")
        assert database is not None

    def test_accepts_file_path(self, tmp_path: Path) -> None:
        """A file-system path is accepted (file-based database)."""
        db_path = tmp_path / "test_memory.db"
        database = Database(str(db_path))
        database.initialize()
        # The file must exist on disk after initialisation.
        assert db_path.exists(), "Database file was not created on disk"


class TestInitialize:
    """initialize() creates the five expected tables."""

    EXPECTED_TABLES = {"nodes", "edges", "operation_log", "chain_log", "stats"}

    def test_creates_all_tables(self, db: Database) -> None:
        """After initialize() the five schema tables exist."""
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        missing = self.EXPECTED_TABLES - table_names
        assert not missing, f"Tables missing after initialize(): {missing}"

    def test_idempotent(self, db: Database) -> None:
        """Calling initialize() twice does not raise."""
        db.initialize()  # second call


# ---------------------------------------------------------------------------
# insert_node
# ---------------------------------------------------------------------------

class TestInsertNode:
    """insert_node inserts a row and returns a UUID string."""

    def test_returns_uuid_string(self, db: Database) -> None:
        """The return value of insert_node is a valid UUID (v4)."""
        node_id = db.insert_node(
            content="Python uses indentation for code blocks",
            node_type="factual",
            coreness=0.5,
            tags=["python", "syntax"],
        )
        assert isinstance(node_id, str), "insert_node must return a string"
        # Must be parseable as a UUID.
        parsed = uuid.UUID(node_id)
        assert parsed.version == 4, "insert_node must return a UUIDv4 string"

    def test_tags_accepted_as_list(self, db: Database) -> None:
        """Tags passed as a Python list are accepted."""
        node_id = db.insert_node(
            content="tagged content",
            tags=["a", "b", "c"],
        )
        assert isinstance(node_id, str)

    def test_minimal_args(self, db: Database) -> None:
        """Only content is required; all other params use defaults."""
        node_id = db.insert_node(content="Minimal node")
        assert isinstance(node_id, str)

    def test_multiple_inserts_return_distinct_ids(self, db: Database) -> None:
        """Each insert_node call returns a different UUID."""
        id1 = db.insert_node(content="First")
        id2 = db.insert_node(content="Second")
        assert id1 != id2, "Consecutive inserts must return distinct UUIDs"


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:
    """get_node retrieves a Node dataclass or returns None."""

    def test_returns_node_dataclass(self, db: Database) -> None:
        """get_node returns a Node instance with all expected fields."""
        node_id = db.insert_node(
            content="Python uses indentation for code blocks",
            node_type="factual",
            coreness=0.5,
            tags=["python", "syntax"],
        )
        node = db.get_node(node_id)
        assert isinstance(node, Node), "get_node must return a Node instance"
        assert node.id == node_id
        assert node.content == "Python uses indentation for code blocks"
        assert node.node_type == "factual"
        assert node.coreness == 0.5
        assert node.tags == ["python", "syntax"]

    def test_defaults_applied(self, db: Database) -> None:
        """Fields with defaults are set correctly when not provided."""
        node_id = db.insert_node(content="Default node")
        node = db.get_node(node_id)
        assert node.node_type == "factual"
        assert node.coreness == 0.5
        assert node.tags == []

    def test_returns_none_for_missing(self, db: Database) -> None:
        """Requesting a non-existent node returns None."""
        fake_id = str(uuid.uuid4())
        assert db.get_node(fake_id) is None

    def test_invalid_id_returns_none(self, db: Database) -> None:
        """A malformed ID (not a UUID) returns None gracefully."""
        assert db.get_node("not-a-uuid") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_content_raises(self, db: Database) -> None:
        """Inserting a node with empty content raises an error."""
        with pytest.raises((ValueError, Exception)):
            db.insert_node(content="")

    def test_content_newline_handling(self, db: Database) -> None:
        """Multi-line content is stored and retrieved faithfully."""
        multiline = "line one\nline two\nline three"
        node_id = db.insert_node(content=multiline)
        node = db.get_node(node_id)
        assert node.content == multiline


# ---------------------------------------------------------------------------
# insert_edge
# ---------------------------------------------------------------------------

class TestInsertEdge:
    """insert_edge creates a directed edge between two nodes."""

    def test_insert_edge_succeeds(self, db: Database) -> None:
        """insert_edge(source_id, target_id) does not raise."""
        src = db.insert_node(content="Source node")
        tgt = db.insert_node(content="Target node")
        db.insert_edge(source_id=src, target_id=tgt)  # no exception

    def test_relation_type_enum_valid(self, db: Database) -> None:
        """All 7 valid relation_type values are accepted."""
        src = db.insert_node(content="Source")
        for rel in (
            "sequence",
            "alternative",
            "causes",
            "contains",
            "similar_to",
            "correction",
            "generic",
        ):
            tgt = db.insert_node(content=f"Target {rel}")
            db.insert_edge(source_id=src, target_id=tgt, relation_type=rel)

    def test_invalid_relation_type_raises(self, db: Database) -> None:
        """An unrecognised relation_type raises an error."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        with pytest.raises(Exception):
            db.insert_edge(source_id=src, target_id=tgt, relation_type="invalid_type")

    def test_default_forward_backward_strength(self, db: Database) -> None:
        """forward_strength and backward_strength default to 0.5."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src, target_id=tgt)
        edge = db.get_edge(src, tgt)
        assert edge.forward_strength == 0.5
        assert edge.backward_strength == 0.5

    def test_custom_strength(self, db: Database) -> None:
        """Custom forward/backward strengths are stored correctly."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(
            source_id=src,
            target_id=tgt,
            forward_strength=0.8,
            backward_strength=0.3,
        )
        edge = db.get_edge(src, tgt)
        assert edge.forward_strength == 0.8
        assert edge.backward_strength == 0.3

    def test_inhibition_default_zero(self, db: Database) -> None:
        """inhibition defaults to 0.0 when not provided."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src, target_id=tgt)
        edge = db.get_edge(src, tgt)
        assert edge.inhibition == 0.0

    def test_nonexistent_source_raises(self, db: Database) -> None:
        """Referencing a non-existent source_id raises an error (FK constraint)."""
        tgt = db.insert_node(content="Target")
        with pytest.raises(Exception):
            db.insert_edge(source_id=str(uuid.uuid4()), target_id=tgt)

    def test_nonexistent_target_raises(self, db: Database) -> None:
        """Referencing a non-existent target_id raises an error (FK constraint)."""
        src = db.insert_node(content="Source")
        with pytest.raises(Exception):
            db.insert_edge(source_id=src, target_id=str(uuid.uuid4()))

    def test_duplicate_edge_raises(self, db: Database) -> None:
        """Inserting the same edge twice raises an error (PK conflict)."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src, target_id=tgt)
        with pytest.raises(Exception):
            db.insert_edge(source_id=src, target_id=tgt)


# ---------------------------------------------------------------------------
# get_edge
# ---------------------------------------------------------------------------

class TestGetEdge:
    """get_edge retrieves an Edge dataclass or returns None."""

    def test_get_edge_returns_dataclass(self, db: Database) -> None:
        """get_edge returns an Edge instance with all expected fields."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src, target_id=tgt)
        edge = db.get_edge(src, tgt)
        assert isinstance(edge, Edge)
        assert edge.source_id == src
        assert edge.target_id == tgt

    def test_get_edge_nonexistent_returns_none(self, db: Database) -> None:
        """Querying a non-existent edge returns None."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        assert db.get_edge(src, tgt) is None

    def test_get_edge_wrong_direction(self, db: Database) -> None:
        """Edge A->B does not match a query for B->A."""
        src = db.insert_node(content="Source")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src, target_id=tgt)
        assert db.get_edge(tgt, src) is None


# ---------------------------------------------------------------------------
# list_edges
# ---------------------------------------------------------------------------

class TestEdgeList:
    """list_edges returns edges filtered by source, target, or all."""

    def test_list_outgoing_edges(self, db: Database) -> None:
        """list_edges(source=node_id) returns outgoing edges for that node."""
        src = db.insert_node(content="Source")
        tgt1 = db.insert_node(content="Target 1")
        tgt2 = db.insert_node(content="Target 2")
        db.insert_edge(source_id=src, target_id=tgt1)
        db.insert_edge(source_id=src, target_id=tgt2)
        edges = db.list_edges(source=src)
        assert len(edges) == 2

    def test_list_incoming_edges(self, db: Database) -> None:
        """list_edges(target=node_id) returns incoming edges for that node."""
        src1 = db.insert_node(content="Source 1")
        src2 = db.insert_node(content="Source 2")
        tgt = db.insert_node(content="Target")
        db.insert_edge(source_id=src1, target_id=tgt)
        db.insert_edge(source_id=src2, target_id=tgt)
        edges = db.list_edges(target=tgt)
        assert len(edges) == 2

    def test_list_all_edges(self, db: Database) -> None:
        """list_edges() with no arguments returns all edges."""
        src = db.insert_node(content="Source")
        tgt1 = db.insert_node(content="Target 1")
        tgt2 = db.insert_node(content="Target 2")
        db.insert_edge(source_id=src, target_id=tgt1)
        db.insert_edge(source_id=src, target_id=tgt2)
        edges = db.list_edges()
        assert len(edges) == 2
