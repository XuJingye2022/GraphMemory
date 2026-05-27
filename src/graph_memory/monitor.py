"""Monitor — read-only component that detects issues and inconsistencies in the graph memory.

The Monitor does not modify memory directly. It returns proposals that the
caller (user or main system) can decide to act on.
"""

from __future__ import annotations

import json

import numpy as np

from graph_memory.core import GraphMemory


class Monitor:
    """Read-only monitor for AI graph memory.

    Parameters
    ----------
    memory : GraphMemory
        The graph memory instance to monitor.
    """

    def __init__(self, memory: GraphMemory):
        self.memory = memory

    # ------------------------------------------------------------------
    # Issue detection (from failed chain logs)
    # ------------------------------------------------------------------

    def detect_issues(self) -> list[dict]:
        """Detect issues from recent failed chain-of-thought logs.

        Steps
        -----
        1. Fetch the last 20 failed chains (``success_only=False``).
        2. For each failed chain, parse ``chain_json`` and locate the last
           node in the ``nodes`` list (the failure point).
        3. For every edge connected to that node, propose ``"mark_dubious"``.
        4. If the chain has an ``error_location`` field, propose ``"reduce"``
           with ``value=0.1`` for that edge.

        Returns
        -------
        list[dict]
            Each dict has keys:

            - **type** – ``"reduce"`` | ``"mark_dubious"`` | ``"add_correction_edge"``
            - **target** – ``"edge:src->tgt"``
            - **reason** – human-readable explanation
            - **value** – (optional) float, present for ``"reduce"`` proposals
        """
        proposals: list[dict] = []
        chains = self.memory.get_chains(limit=20, success_only=False)

        for chain in chains:
            try:
                chain_data = json.loads(chain["chain_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            nodes: list = chain_data.get("nodes", [])
            if not nodes:
                continue

            last_node = nodes[-1]

            # Propose mark_dubious for every edge associated with the failure
            # node (both outgoing and incoming).
            for edge in self.memory.db.list_edges(source=last_node):
                proposals.append({
                    "type": "mark_dubious",
                    "target": f"edge:{edge.source_id}->{edge.target_id}",
                    "reason": (
                        f"Chain failed at this edge: "
                        f"{edge.source_id}->{edge.target_id}"
                    ),
                })
            for edge in self.memory.db.list_edges(target=last_node):
                proposals.append({
                    "type": "mark_dubious",
                    "target": f"edge:{edge.source_id}->{edge.target_id}",
                    "reason": (
                        f"Chain failed at this edge: "
                        f"{edge.source_id}->{edge.target_id}"
                    ),
                })

            # If an explicit error_location is recorded, propose a reduction.
            error_loc = chain.get("error_location")
            if error_loc:
                proposals.append({
                    "type": "reduce",
                    "target": error_loc,
                    "reason": f"Chain failed at this edge: {error_loc}",
                    "value": 0.1,
                })

        return proposals

    # ------------------------------------------------------------------
    # Consistency checking
    # ------------------------------------------------------------------

    def check_consistency(self) -> list[dict]:
        """Check the graph for structural consistency issues.

        Detects:

        - **duplicate**: two nodes whose ``content`` is identical.
        - **stale**: an edge whose endpoints have embedding similarity > 0.95
          but whose ``relation_type`` is not ``"similar_to"``.
        - (contradiction detection is reserved for future work).

        Returns
        -------
        list[dict]
            Each dict has keys:

            - **type** – ``"contradiction"`` | ``"duplicate"`` | ``"stale"``
            - **node_a** – UUID of the first node involved
            - **node_b** – UUID of the second node involved
            - **description** – human-readable explanation
        """
        conflicts: list[dict] = []
        nodes = self.memory.db.list_nodes(include_deleted=False)

        # -- Duplicate detection -------------------------------------------
        seen: dict[str, str] = {}
        for node in nodes:
            if node.content in seen:
                conflicts.append({
                    "type": "duplicate",
                    "node_a": seen[node.content],
                    "node_b": node.id,
                    "description": (
                        f"Nodes have identical content: "
                        f"{node.content[:50]}{'...' if len(node.content) > 50 else ''}"
                    ),
                })
            else:
                seen[node.content] = node.id

        # -- Stale edge detection ------------------------------------------
        edges = self.memory.db.list_edges()
        for edge in edges:
            if edge.relation_type == "similar_to":
                continue

            src_node = self.memory.get_node(edge.source_id)
            tgt_node = self.memory.get_node(edge.target_id)
            if src_node is None or tgt_node is None:
                continue
            if src_node.embedding is None or tgt_node.embedding is None:
                continue

            try:
                src_vec = np.frombuffer(src_node.embedding, dtype=np.float32)
                tgt_vec = np.frombuffer(tgt_node.embedding, dtype=np.float32)
                sim = float(
                    np.dot(src_vec, tgt_vec)
                    / (np.linalg.norm(src_vec) * np.linalg.norm(tgt_vec))
                )
                if sim > 0.95:
                    conflicts.append({
                        "type": "stale",
                        "node_a": edge.source_id,
                        "node_b": edge.target_id,
                        "description": (
                            f"Edge relation_type is '{edge.relation_type}' "
                            f"but nodes have cosine similarity "
                            f"{sim:.3f} > 0.95"
                        ),
                    })
            except Exception:
                continue

        return conflicts
