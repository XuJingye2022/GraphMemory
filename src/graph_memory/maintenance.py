"""Maintenance operations for the AI memory system.

Provides merge, prune, and replay functions that operate on a
:class:`GraphMemory` instance to keep the graph healthy, compact,
and aligned with the agent's current role.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from graph_memory.core import GraphMemory


# ---------------------------------------------------------------------------
# merge_similar_nodes
# ---------------------------------------------------------------------------


def merge_similar_nodes(
    memory: GraphMemory,
    similarity_threshold: float = 0.9,
) -> dict[str, int]:
    """Merge factual nodes whose embeddings are similar above *similarity_threshold*.

    For each pair of non-deleted factual nodes whose cosine similarity
    exceeds *similarity_threshold*:

    1. A new merged node is created (content = ``"A | B"``,
       coreness = max of the two).
    2. Every edge pointing to either old node is redirected to the new
       node, keeping the original strengths.
    3. Both original nodes are soft-deleted.

    Parameters
    ----------
    memory : GraphMemory
        The memory instance to operate on.
    similarity_threshold : float
        Cosine similarity threshold (default ``0.9``).

    Returns
    -------
    dict[str, int]
        ``{"merged_pairs": int, "nodes_removed": int}``.
    """
    if memory.engine is None:
        return {"merged_pairs": 0, "nodes_removed": 0}

    all_nodes = memory.db.list_nodes(include_deleted=False)
    factual_nodes = [n for n in all_nodes if n.node_type == "factual"]

    merged_pairs = 0
    nodes_removed = 0
    processed: set[str] = set()

    for i in range(len(factual_nodes)):
        ni = factual_nodes[i]
        if ni.id in processed:
            continue
        for j in range(i + 1, len(factual_nodes)):
            nj = factual_nodes[j]
            if nj.id in processed:
                continue
            if ni.embedding is None or nj.embedding is None:
                continue

            vec_i = np.frombuffer(ni.embedding, dtype=np.float32).tolist()
            vec_j = np.frombuffer(nj.embedding, dtype=np.float32).tolist()
            sim = memory.engine.similarity(vec_i, vec_j)

            if sim > similarity_threshold:
                merged_content = f"{ni.content} | {nj.content}"
                merged_coreness = max(ni.coreness, nj.coreness)
                merged_tags = list(set(ni.tags + nj.tags))

                new_id = memory.add_node(
                    content=merged_content,
                    node_type="factual",
                    coreness=merged_coreness,
                    tags=merged_tags,
                )

                for old_id in (ni.id, nj.id):
                    _redirect_edges(memory.db, old_id, new_id)
                    memory.db.soft_delete_node(old_id)

                processed.add(ni.id)
                processed.add(nj.id)
                merged_pairs += 1
                nodes_removed += 2

    return {"merged_pairs": merged_pairs, "nodes_removed": nodes_removed}


def _redirect_edges(db: Any, old_id: str, new_id: str) -> None:
    """Redirect all edges that reference *old_id* to *new_id*.

    Both outgoing and incoming edges are redirected.
    Duplicate edges (e.g. when both old nodes shared a neighbour) are
    silently skipped.
    """
    for edge in db.list_edges(source=old_id):
        try:
            db.insert_edge(
                source_id=new_id,
                target_id=edge.target_id,
                forward_strength=edge.forward_strength,
                backward_strength=edge.backward_strength,
                relation_type=edge.relation_type,
                inhibition=edge.inhibition,
            )
        except Exception:
            pass

    for edge in db.list_edges(target=old_id):
        try:
            db.insert_edge(
                source_id=edge.source_id,
                target_id=new_id,
                forward_strength=edge.forward_strength,
                backward_strength=edge.backward_strength,
                relation_type=edge.relation_type,
                inhibition=edge.inhibition,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# prune_dead_nodes
# ---------------------------------------------------------------------------


def prune_dead_nodes(
    memory: GraphMemory,
    coreness_threshold: float = 0.2,
    days_threshold: int = 90,
) -> dict[str, int]:
    """Prune low-utility nodes from the graph.

    Two-phase pruning:

    1. **Soft delete** active nodes whose ``coreness < coreness_threshold``
       and that have **zero** edges (neither incoming nor outgoing).
    2. **Hard delete** already-soft-deleted nodes that still have zero
       edges and whose ``last_accessed`` is older than *days_threshold*.

    Parameters
    ----------
    memory : GraphMemory
        The memory instance to operate on.
    coreness_threshold : float
        Maximum coreness for a node to be considered dead (default ``0.2``).
    days_threshold : int
        Number of days after which a soft-deleted node is eligible for
        hard deletion (default ``90``).

    Returns
    -------
    dict[str, int]
        ``{"soft_deleted": int, "hard_deleted": int}``.
    """
    soft_deleted = 0
    hard_deleted = 0

    # ---- Phase 1: soft-delete low-coreness, edgeless nodes ----------------
    all_nodes = memory.db.list_nodes(include_deleted=False)
    for node in all_nodes:
        if node.coreness >= coreness_threshold:
            continue
        if _has_any_edge(memory.db, node.id):
            continue
        memory.db.soft_delete_node(node.id)
        soft_deleted += 1

    # ---- Phase 2: hard-delete old, already-soft-deleted, edgeless nodes ----
    cutoff = (datetime.now() - timedelta(days=days_threshold)).isoformat()
    all_nodes_incl_deleted = memory.db.list_nodes(include_deleted=True)
    for node in all_nodes_incl_deleted:
        if not node.is_deleted:
            continue
        if _has_any_edge(memory.db, node.id):
            continue
        if node.last_accessed >= cutoff:
            continue
        memory.db._conn.execute(
            "DELETE FROM nodes WHERE id = ?", (node.id,)
        )
        memory.db._conn.commit()
        hard_deleted += 1

    return {"soft_deleted": soft_deleted, "hard_deleted": hard_deleted}


def _has_any_edge(db: Any, node_id: str) -> bool:
    """Return ``True`` if *node_id* has at least one incoming or outgoing edge."""
    if db.list_edges(source=node_id):
        return True
    if db.list_edges(target=node_id):
        return True
    return False


# ---------------------------------------------------------------------------
# replay_core_paths
# ---------------------------------------------------------------------------


def replay_core_paths(
    memory: GraphMemory,
    agent_profile: dict[str, Any],
) -> dict[str, int]:
    """Strengthen nodes and edges that match the agent's current profile.

    Nodes whose tags intersect with ``agent_profile["roles"]`` **and**
    whose ``coreness > 0.6`` receive a small coreness boost (``+0.005``,
    capped at ``1.0``).  Their outgoing edges also get a base-stability
    boost (``+0.01``, capped at ``0.95``).

    Parameters
    ----------
    memory : GraphMemory
        The memory instance to operate on.
    agent_profile : dict
        Must contain at least a ``"roles"`` key (``list[str]``).  A
        ``"core_tasks"`` key is accepted but **not** used in the current
        implementation.

    Returns
    -------
    dict[str, int]
        ``{"nodes_strengthened": int, "edges_strengthened": int}``.
    """
    roles = set(agent_profile.get("roles", []))
    if not roles:
        return {"nodes_strengthened": 0, "edges_strengthened": 0}

    nodes_strengthened = 0
    edges_strengthened = 0

    for node in memory.db.list_nodes(include_deleted=False):
        node_tags = set(node.tags)
        if not node_tags.intersection(roles):
            continue
        if node.coreness <= 0.6:
            continue

        new_coreness = min(node.coreness + 0.005, 1.0)
        memory.db.update_node_coreness(node.id, new_coreness)
        nodes_strengthened += 1

        for edge in memory.db.list_edges(source=node.id):
            new_stability = min(edge.base_stability + 0.01, 0.95)
            memory.db.update_edge_stability(
                edge.source_id, edge.target_id, new_stability,
            )
            edges_strengthened += 1

    return {
        "nodes_strengthened": nodes_strengthened,
        "edges_strengthened": edges_strengthened,
    }
