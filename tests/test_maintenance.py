"""Tests for graph_memory.maintenance — RED phase.

Covers merge_similar_nodes, prune_dead_nodes, and replay_core_paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from graph_memory import GraphMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embeddings_available() -> bool:
    """Return True if EmbeddingEngine can be instantiated (model downloaded)."""
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
# merge_similar_nodes
# ---------------------------------------------------------------------------


class TestMergeSimilarNodes:
    """merge_similar_nodes merges factual nodes whose embeddings are close."""

    @pytest.mark.skipif(
        not _embeddings_available(),
        reason="EmbeddingEngine model not downloaded -- skip merge test",
    )
    def test_merge_similar_nodes(self, mem: GraphMemory) -> None:
        """Two semantically similar factual nodes get merged."""
        # pylint: disable=import-outside-toplevel
        from graph_memory.maintenance import merge_similar_nodes

        n1 = mem.add_node(content="Python programming language")
        n2 = mem.add_node(content="Python is used for coding")

        result = merge_similar_nodes(mem, similarity_threshold=0.5)

        assert result["merged_pairs"] >= 1
        assert result["nodes_removed"] >= 2

        # Both original nodes should be soft-deleted
        node1 = mem.get_node(n1)
        node2 = mem.get_node(n2)
        assert node1 is None or node1.is_deleted is True
        assert node2 is None or node2.is_deleted is True

        # A new merged node should exist in the active set
        remaining = [
            n
            for n in mem.db.list_nodes(include_deleted=False)
            if n.id not in (n1, n2)
        ]
        assert len(remaining) >= 1

    @pytest.mark.skipif(
        not _embeddings_available(),
        reason="EmbeddingEngine model not downloaded -- skip merge test",
    )
    def test_no_merge_for_dissimilar_nodes(self, mem: GraphMemory) -> None:
        """Dissimilar nodes should not be merged."""
        from graph_memory.maintenance import merge_similar_nodes

        n1 = mem.add_node(content="Python programming language")
        n2 = mem.add_node(content="Chocolate cake recipe")

        result = merge_similar_nodes(mem, similarity_threshold=0.9)

        assert result["merged_pairs"] == 0

    def test_merge_returns_zero_when_no_engine(self, mem: GraphMemory) -> None:
        """When EmbeddingEngine is unavailable, merge returns zeroes (no crash)."""
        # Force engine to None to simulate the unavailability path.
        if mem.engine is not None:
            pytest.skip("Engine is available -- cannot test unavailable path")

        from graph_memory.maintenance import merge_similar_nodes

        result = merge_similar_nodes(mem, similarity_threshold=0.9)
        assert result == {"merged_pairs": 0, "nodes_removed": 0}


# ---------------------------------------------------------------------------
# prune_dead_nodes
# ---------------------------------------------------------------------------


class TestPruneDeadNodes:
    """prune_dead_nodes removes low-utility nodes from the graph."""

    def test_soft_delete_low_coreness_isolated(self, mem: GraphMemory) -> None:
        """Low-coreness isolated nodes get soft-deleted."""
        from graph_memory.maintenance import prune_dead_nodes

        nid = mem.add_node(content="Orphan node", coreness=0.1)

        result = prune_dead_nodes(mem, coreness_threshold=0.2, days_threshold=90)

        assert result["soft_deleted"] >= 1
        node = mem.get_node(nid)
        assert node is None or node.is_deleted is True

    def test_high_coreness_not_pruned(self, mem: GraphMemory) -> None:
        """High-coreness nodes should NOT be soft-deleted."""
        from graph_memory.maintenance import prune_dead_nodes

        nid = mem.add_node(content="Important node", coreness=0.8)

        result = prune_dead_nodes(mem, coreness_threshold=0.2, days_threshold=90)

        assert result["soft_deleted"] == 0
        node = mem.get_node(nid)
        assert node is not None and node.is_deleted is False

    def test_low_coreness_with_edges_not_pruned(self, mem: GraphMemory) -> None:
        """Low-coreness nodes with at least one edge should NOT be soft-deleted."""
        from graph_memory.maintenance import prune_dead_nodes

        n1 = mem.add_node(content="Hub node", coreness=0.1)
        n2 = mem.add_node(content="Connected node", coreness=0.8)
        mem.add_edge(source_id=n1, target_id=n2)

        result = prune_dead_nodes(mem, coreness_threshold=0.2, days_threshold=90)

        assert result["soft_deleted"] == 0
        node = mem.get_node(n1)
        assert node is not None and node.is_deleted is False

    def test_hard_deletes_old_soft_deleted(self, mem: GraphMemory) -> None:
        """Already-soft-deleted, edgeless nodes older than days_threshold get
        hard-deleted."""
        from graph_memory.maintenance import prune_dead_nodes

        nid = mem.add_node(content="Old deleted node", coreness=0.1)
        mem.db.soft_delete_node(nid)

        # Backdate last_accessed to 100 days ago
        old_ts = (datetime.now() - timedelta(days=100)).isoformat()
        mem.db._conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?",
            (old_ts, nid),
        )
        mem.db._conn.commit()

        result = prune_dead_nodes(mem, coreness_threshold=0.2, days_threshold=90)

        assert result["hard_deleted"] >= 1
        # Node should no longer exist in the database
        node = mem.get_node(nid)
        assert node is None


# ---------------------------------------------------------------------------
# replay_core_paths
# ---------------------------------------------------------------------------


class TestReplayCorePaths:
    """replay_core_paths strengthens nodes matching an agent's profile."""

    def test_strengthens_matching_nodes(self, mem: GraphMemory) -> None:
        """Nodes whose tags match agent roles get a coreness boost."""
        from graph_memory.maintenance import replay_core_paths

        nid = mem.add_node(
            content="SQL query optimization",
            tags=["SQL"],
            coreness=0.7,
        )
        agent_profile = {
            "roles": ["数据分析", "SQL"],
            "core_tasks": ["编写 SQL 查询"],
        }

        before = mem.get_node(nid).coreness
        result = replay_core_paths(mem, agent_profile)
        after = mem.get_node(nid).coreness

        assert after > before
        assert result["nodes_strengthened"] >= 1

    def test_ignores_non_matching_nodes(self, mem: GraphMemory) -> None:
        """Nodes with no matching tags should NOT be strengthened."""
        from graph_memory.maintenance import replay_core_paths

        nid = mem.add_node(
            content="Weather forecast",
            tags=["weather"],
            coreness=0.7,
        )
        agent_profile = {
            "roles": ["数据分析", "SQL"],
            "core_tasks": ["编写 SQL 查询"],
        }

        before = mem.get_node(nid).coreness
        result = replay_core_paths(mem, agent_profile)

        after = mem.get_node(nid).coreness
        assert after == before
        assert result["nodes_strengthened"] == 0

    def test_ignores_low_coreness_nodes(self, mem: GraphMemory) -> None:
        """Nodes with matching tags but coreness <= 0.6 should NOT be
        strengthened."""
        from graph_memory.maintenance import replay_core_paths

        nid = mem.add_node(
            content="Basic SQL knowledge",
            tags=["SQL"],
            coreness=0.3,
        )
        agent_profile = {
            "roles": ["数据分析", "SQL"],
            "core_tasks": ["编写 SQL 查询"],
        }

        before = mem.get_node(nid).coreness
        result = replay_core_paths(mem, agent_profile)

        after = mem.get_node(nid).coreness
        assert after == before
        assert result["nodes_strengthened"] == 0

    def test_strengthens_outgoing_edges(self, mem: GraphMemory) -> None:
        """Outgoing edges of matching nodes get a base_stability boost."""
        from graph_memory.maintenance import replay_core_paths

        n1 = mem.add_node(content="SQL basics", tags=["SQL"], coreness=0.7)
        n2 = mem.add_node(content="JOIN syntax", tags=["SQL"], coreness=0.5)
        mem.add_edge(source_id=n1, target_id=n2, forward_strength=0.5)

        edge_before = mem.get_edge(n1, n2)
        replay_core_paths(
            mem,
            {"roles": ["SQL"], "core_tasks": []},
        )
        edge_after = mem.get_edge(n1, n2)

        assert edge_after.base_stability > edge_before.base_stability

    def test_empty_roles_returns_zeros(self, mem: GraphMemory) -> None:
        """An agent_profile with empty roles list should strengthen nothing."""
        from graph_memory.maintenance import replay_core_paths

        mem.add_node(content="Anything", tags=["x"], coreness=0.8)

        result = replay_core_paths(mem, {"roles": [], "core_tasks": []})
        assert result == {"nodes_strengthened": 0, "edges_strengthened": 0}
