"""Tests for graph_memory.monitor — RED phase.

Covers ``detect_issues`` and ``check_consistency``.

Usage::

    pytest tests/test_monitor.py -v
"""

from __future__ import annotations

import pytest

from graph_memory import GraphMemory
from graph_memory.monitor import Monitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem() -> GraphMemory:
    """Return a fresh in-memory GraphMemory instance."""
    return GraphMemory(":memory:")


@pytest.fixture
def monitor(mem: GraphMemory) -> Monitor:
    """Return a Monitor bound to the fresh in-memory graph."""
    return Monitor(mem)


# ---------------------------------------------------------------------------
# detect_issues
# ---------------------------------------------------------------------------


class TestDetectIssues:
    """detect_issues finds proposals from failed chain logs."""

    def test_detect_issues_from_failed_chain(
        self, mem: GraphMemory, monitor: Monitor
    ) -> None:
        """A failed chain with edges should produce mark_dubious proposals."""
        n1 = mem.add_node("step 1")
        n2 = mem.add_node("step 2")
        mem.add_edge(source_id=n1, target_id=n2)

        mem.log_chain(
            node_ids=[n1, n2],
            edge_ids=[(n1, n2)],
            query="test query",
            success=False,
        )

        proposals = monitor.detect_issues()
        assert len(proposals) > 0

        # Last node is n2, which has an incoming edge n1->n2
        dubious = [p for p in proposals if p["type"] == "mark_dubious"]
        assert len(dubious) >= 1
        assert f"edge:{n1}->{n2}" in [d["target"] for d in dubious]

    def test_no_issues_when_no_failures(self, monitor: Monitor) -> None:
        """No failed chains should produce empty proposals."""
        proposals = monitor.detect_issues()
        assert proposals == []

    def test_detect_issues_with_error_location(
        self, mem: GraphMemory, monitor: Monitor
    ) -> None:
        """Error location in chain log should produce a reduce proposal."""
        n1 = mem.add_node("step 1")
        n2 = mem.add_node("step 2")
        mem.add_edge(source_id=n1, target_id=n2)

        # ``log_chain`` always sets ``error_location=None``, so we insert
        # directly via the database layer to set a custom error location.
        mem.db.insert_chain_log(
            chain_json=(
                '{"nodes": ["' + n1 + '", "' + n2 + '"], '
                '"edges": [["' + n1 + '", "' + n2 + '"]], '
                '"query": "test"}'
            ),
            success=0,
            error_location=f"edge:{n1}->{n2}",
        )

        proposals = monitor.detect_issues()
        reduces = [p for p in proposals if p["type"] == "reduce"]
        assert len(reduces) >= 1
        assert reduces[0]["target"] == f"edge:{n1}->{n2}"
        assert reduces[0]["value"] == 0.1


# ---------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------


class TestCheckConsistency:
    """check_consistency finds conflicts in the graph."""

    def test_detect_duplicate_content(
        self, mem: GraphMemory, monitor: Monitor
    ) -> None:
        """Two nodes with identical content should be flagged as duplicates."""
        n1 = mem.add_node("duplicate content")
        n2 = mem.add_node("duplicate content")

        conflicts = monitor.check_consistency()
        duplicates = [c for c in conflicts if c["type"] == "duplicate"]
        assert len(duplicates) >= 1
        ids_in_conflict = {duplicates[0]["node_a"], duplicates[0]["node_b"]}
        assert {n1, n2} == ids_in_conflict

    def test_no_duplicates_for_different_content(
        self, mem: GraphMemory, monitor: Monitor
    ) -> None:
        """Nodes with different content should not be flagged."""
        mem.add_node("content A")
        mem.add_node("content B")

        conflicts = monitor.check_consistency()
        duplicates = [c for c in conflicts if c["type"] == "duplicate"]
        assert len(duplicates) == 0

    def test_no_conflicts_when_empty(self, monitor: Monitor) -> None:
        """Empty graph should produce no conflicts."""
        conflicts = monitor.check_consistency()
        assert conflicts == []
