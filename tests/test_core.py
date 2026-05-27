"""Tests for GraphMemory — core public interface (RED phase).

GraphMemory is the facade that composes Database, EmbeddingEngine, and
reinforcement-logic.  These tests exercise only the public API so the
internal composition is free to change.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from graph_memory import GraphMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embeddings_available() -> bool:
    """Return True if EmbeddingEngine can be instantiated (model is downloaded)."""
    try:
        from graph_memory.embeddings import EmbeddingEngine

        EmbeddingEngine()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem() -> GraphMemory:
    """Return a fresh in-memory GraphMemory instance."""
    return GraphMemory(":memory:")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGraphMemoryConstruction:
    """GraphMemory(path) creates a memory store ready for use."""

    def test_constructs_with_memory_db(self) -> None:
        """GraphMemory(':memory:') constructs without error."""
        memory = GraphMemory(":memory:")
        assert memory is not None

    def test_constructs_with_file_path(self, tmp_path: Path) -> None:
        """A file-system path is accepted (file-based database)."""
        db_path = tmp_path / "test_memory.db"
        memory = GraphMemory(str(db_path))
        assert memory is not None


# ---------------------------------------------------------------------------
# add_node / get_node
# ---------------------------------------------------------------------------

class TestAddNode:
    """add_node stores a node and returns its ID."""

    def test_add_node_returns_uuid(self, mem: GraphMemory) -> None:
        """add_node returns a valid UUID v4 string."""
        node_id = mem.add_node(content="Python uses indentation")
        assert isinstance(node_id, str)
        parsed = uuid.UUID(node_id)
        assert parsed.version == 4

    def test_add_node_stores_content(self, mem: GraphMemory) -> None:
        """Added content is retrievable via get_node with all fields intact."""
        node_id = mem.add_node(
            content="Python uses indentation",
            node_type="factual",
            coreness=0.5,
            tags=["python"],
        )
        node = mem.get_node(node_id)
        assert node is not None
        assert node.content == "Python uses indentation"
        assert node.node_type == "factual"
        assert node.coreness == 0.5
        assert node.tags == ["python"]

    @pytest.mark.skipif(
        not _embeddings_available(),
        reason="EmbeddingEngine model not downloaded -- skip embedding test",
    )
    def test_add_node_computes_embedding(self, mem: GraphMemory) -> None:
        """A factual node gets an embedding computed automatically."""
        node_id = mem.add_node(content="Python uses indentation", node_type="factual")
        node = mem.get_node(node_id)
        assert node is not None
        assert node.embedding is not None, (
            "Factual nodes should have a non-None embedding"
        )

    def test_add_node_no_embedding_for_procedural(self, mem: GraphMemory) -> None:
        """A procedural node does not get an embedding (pure steps do not need semantic retrieval)."""
        node_id = mem.add_node(
            content="To install Python, download from python.org",
            node_type="procedural",
        )
        node = mem.get_node(node_id)
        assert node is not None
        assert node.embedding is None, (
            "Procedural nodes should not have an embedding"
        )

    def test_add_node_empty_content_raises(self, mem: GraphMemory) -> None:
        """Empty content raises a ValueError."""
        with pytest.raises((ValueError, Exception)):
            mem.add_node(content="")


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------

class TestAddEdge:
    """add_edge creates a directed edge between two nodes."""

    def test_add_edge_between_nodes(self, mem: GraphMemory) -> None:
        """Creating an edge between two valid nodes succeeds."""
        src = mem.add_node(content="Source node")
        tgt = mem.add_node(content="Target node")
        mem.add_edge(source_id=src, target_id=tgt, relation_type="sequence")
        # No exception means success.

    def test_add_edge_nonexistent_node_raises(self, mem: GraphMemory) -> None:
        """Referencing a non-existent source or target node raises an error."""
        real = mem.add_node(content="Real node")
        fake = str(uuid.uuid4())
        with pytest.raises(Exception):
            mem.add_edge(source_id=real, target_id=fake)
        with pytest.raises(Exception):
            mem.add_edge(source_id=fake, target_id=real)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecall:
    """recall performs semantic search over stored nodes."""

    def test_recall_returns_list(self, mem: GraphMemory) -> None:
        """recall returns a list of (Node, float) tuples."""
        mem.add_node(content="Python programming")
        results = mem.recall("python")
        assert isinstance(results, list)

    def test_recall_empty_db(self, mem: GraphMemory) -> None:
        """recall on an empty database returns an empty list."""
        results = mem.recall("anything")
        assert results == []

    @pytest.mark.skipif(
        not _embeddings_available(),
        reason="EmbeddingEngine model not downloaded -- skip recall ranking test",
    )
    def test_recall_returns_related_nodes(self, mem: GraphMemory) -> None:
        """Nodes relevant to the query are ranked above irrelevant ones."""
        mem.add_node(content="Python programming language", tags=["python"])
        mem.add_node(content="Weather forecast for today", tags=["weather"])
        results = mem.recall("coding language", top_k=5)
        assert len(results) >= 1
        top_content = results[0][0].content
        assert "python" in top_content.lower() or "programming" in top_content.lower()

    def test_recall_respects_top_k(self, mem: GraphMemory) -> None:
        """top_k limits the number of returned results."""
        for i in range(5):
            mem.add_node(content=f"Concept number {i}")
        results = mem.recall("concept", top_k=2)
        assert len(results) <= 2

    def test_recall_respects_tags_filter(self, mem: GraphMemory) -> None:
        """tags filter restricts results to nodes whose tags include the given value."""
        mem.add_node(content="Python tutorial", tags=["python", "tutorial"])
        mem.add_node(content="Java tutorial", tags=["java", "tutorial"])
        mem.add_node(content="Weather report", tags=["weather"])
        results = mem.recall("tutorial", tags=["python"])
        for node, _score in results:
            assert "python" in node.tags, (
                f"Expected 'python' in node.tags, got {node.tags}"
            )


# ---------------------------------------------------------------------------
# reinforce
# ---------------------------------------------------------------------------

class TestReinforce:
    """reinforce_path increases coreness along a reasoning chain."""

    def test_reinforce_increases_coreness(self, mem: GraphMemory) -> None:
        """Coreness of nodes in the path increases after reinforcement."""
        a = mem.add_node(content="Node A", coreness=0.5)
        b = mem.add_node(content="Node B", coreness=0.5)
        c = mem.add_node(content="Node C", coreness=0.5)
        mem.add_edge(source_id=a, target_id=b)
        mem.add_edge(source_id=b, target_id=c)

        before = mem.get_node(a).coreness
        mem.reinforce_path(node_ids=[a, b, c], edge_ids=[(a, b), (b, c)])
        after = mem.get_node(a).coreness
        assert after > before

    def test_reinforce_first_node_gets_most(self, mem: GraphMemory) -> None:
        """The first node in the path receives the largest coreness boost."""
        a = mem.add_node(content="Node A", coreness=0.5)
        b = mem.add_node(content="Node B", coreness=0.5)
        c = mem.add_node(content="Node C", coreness=0.5)
        mem.add_edge(source_id=a, target_id=b)
        mem.add_edge(source_id=b, target_id=c)

        mem.reinforce_path(node_ids=[a, b, c], edge_ids=[(a, b), (b, c)])

        delta_a = mem.get_node(a).coreness - 0.5
        delta_b = mem.get_node(b).coreness - 0.5
        delta_c = mem.get_node(c).coreness - 0.5

        assert delta_a > delta_b, (
            f"First node boost ({delta_a}) should exceed second node boost ({delta_b})"
        )


# ---------------------------------------------------------------------------
# correct_error — 错误修正
# ---------------------------------------------------------------------------

class TestCorrectError:
    """correct_error handles 5 correction types with permission levels."""

    def test_reduce_edge_strength(self, mem: GraphMemory) -> None:
        """reduce lowers base_stability by the given value (capped at 20%)."""
        src = mem.add_node(content="Source")
        tgt = mem.add_node(content="Target")
        mem.add_edge(source_id=src, target_id=tgt)
        edge = mem.get_edge(src, tgt)
        original = edge.base_stability

        mem.correct_error(
            location=f"edge:{src}->{tgt}",
            correction_type="reduce",
            value=0.15,
        )

        edge_after = mem.get_edge(src, tgt)
        expected = max(0.0, original - 0.15)
        assert edge_after.base_stability == expected, (
            f"Expected base_stability {expected}, got {edge_after.base_stability}"
        )

    def test_add_correction_edge(self, mem: GraphMemory) -> None:
        """add_correction_edge creates a correction edge with high strength."""
        src = mem.add_node(content="Error source")
        wrong = mem.add_node(content="Wrong target")
        correct = mem.add_node(content="Correct target")
        mem.add_edge(source_id=src, target_id=wrong)

        mem.correct_error(
            location=f"edge:{src}->{wrong}",
            correction_type="add_correction_edge",
            correct_target_id=correct,
        )

        correction_edge = mem.get_edge(src, correct)
        assert correction_edge is not None, "Correction edge should exist"
        assert correction_edge.relation_type == "correction"
        assert correction_edge.forward_strength >= 0.7, (
            f"Expected forward_strength >= 0.7, got {correction_edge.forward_strength}"
        )

    def test_inhibit_edge(self, mem: GraphMemory) -> None:
        """inhibit sets the edge's inhibition to the specified value."""
        src = mem.add_node(content="Source")
        tgt = mem.add_node(content="Target")
        mem.add_edge(source_id=src, target_id=tgt)

        mem.correct_error(
            location=f"edge:{src}->{tgt}",
            correction_type="inhibit",
            value=0.8,
        )

        edge = mem.get_edge(src, tgt)
        assert edge.inhibition == 0.8, (
            f"Expected inhibition 0.8, got {edge.inhibition}"
        )

    def test_mark_dubious(self, mem: GraphMemory) -> None:
        """mark_dubious sets the edge's is_dubious flag to True."""
        src = mem.add_node(content="Source")
        tgt = mem.add_node(content="Target")
        mem.add_edge(source_id=src, target_id=tgt)

        mem.correct_error(
            location=f"edge:{src}->{tgt}",
            correction_type="mark_dubious",
        )

        edge = mem.get_edge(src, tgt)
        assert edge.is_dubious is True, (
            "Edge should be marked as dubious"
        )

    def test_delete_node(self, mem: GraphMemory) -> None:
        """delete soft-deletes a node (is_deleted=True)."""
        node_id = mem.add_node(content="To be deleted")

        mem.correct_error(
            location=f"node:{node_id}",
            correction_type="delete",
        )

        node = mem.get_node(node_id)
        assert node is not None
        assert node.is_deleted is True, (
            "Node should be soft-deleted"
        )

    def test_correction_levels(self, mem: GraphMemory) -> None:
        """auto_approve=False raises PermissionError for all correction types."""
        src = mem.add_node(content="Source")
        tgt = mem.add_node(content="Target")
        mem.add_edge(source_id=src, target_id=tgt)

        with pytest.raises(PermissionError, match="Manual approval required"):
            mem.correct_error(
                location=f"edge:{src}->{tgt}",
                correction_type="reduce",
                value=0.1,
                auto_approve=False,
            )


# ---------------------------------------------------------------------------
# log_chain / get_chains — 思维链记录
# ---------------------------------------------------------------------------

class TestChainLog:
    """log_chain stores reasoning chains; get_chains retrieves them."""

    def test_log_chain_success(self, mem: GraphMemory) -> None:
        """A successful chain is stored and retrievable."""
        a = mem.add_node(content="Step A")
        b = mem.add_node(content="Step B")
        c = mem.add_node(content="Step C")
        mem.add_edge(source_id=a, target_id=b)
        mem.add_edge(source_id=b, target_id=c)

        mem.log_chain(
            node_ids=[a, b, c],
            edge_ids=[(a, b), (b, c)],
            query="test query",
            success=True,
        )

        chains = mem.get_chains(limit=10)
        assert len(chains) >= 1

    def test_log_chain_failure(self, mem: GraphMemory) -> None:
        """A failed chain is also stored correctly."""
        a = mem.add_node(content="Wrong step")
        b = mem.add_node(content="Dead end")
        mem.add_edge(source_id=a, target_id=b)

        mem.log_chain(
            node_ids=[a, b],
            edge_ids=[(a, b)],
            query="failing query",
            success=False,
        )

        chains = mem.get_chains(limit=10, success_only=False)
        assert len(chains) >= 1

    def test_get_chains_returns_limited_results(self, mem: GraphMemory) -> None:
        """get_chains(limit=10) returns up to 10 recent chain logs."""
        a = mem.add_node(content="Root")
        for i in range(3):
            b = mem.add_node(content=f"Branch {i}")
            mem.add_edge(source_id=a, target_id=b)
            mem.log_chain(
                node_ids=[a, b],
                edge_ids=[(a, b)],
                query=f"query {i}",
                success=True,
            )

        chains = mem.get_chains(limit=10)
        assert isinstance(chains, list)
        assert len(chains) >= 1

    def test_get_chains_by_success(self, mem: GraphMemory) -> None:
        """get_chains(success_only=True) returns only successful chains."""
        a = mem.add_node(content="Root")
        b_ok = mem.add_node(content="Good path")
        c_fail = mem.add_node(content="Bad path")
        mem.add_edge(source_id=a, target_id=b_ok)
        mem.add_edge(source_id=a, target_id=c_fail)

        mem.log_chain(
            node_ids=[a, b_ok],
            edge_ids=[(a, b_ok)],
            query="good",
            success=True,
        )
        mem.log_chain(
            node_ids=[a, c_fail],
            edge_ids=[(a, c_fail)],
            query="bad",
            success=False,
        )

        chains = mem.get_chains(success_only=True)
        assert len(chains) >= 1
        for chain in chains:
            success = chain["success"] if isinstance(chain, dict) else chain.success
            assert success is True, (
                "Only successful chains should be returned"
            )
