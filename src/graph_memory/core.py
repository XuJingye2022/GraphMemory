"""GraphMemory — facade that composes Database, EmbeddingEngine, and reinforcement logic."""

import json
from typing import Optional

import numpy as np

from .config import Config
from .db import Database
from .embeddings import EmbeddingEngine
from .models import Edge, Node


class GraphMemory:
    """Public API for the AI memory system.

    Composes Database, EmbeddingEngine, and reinforcement logic into a single
    facade that the GREEN phase tests exercise.
    """

    def __init__(self, db_path: str = ":memory:", config: Optional[Config] = None):
        """Initialise database + embedding engine.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database, or ``:memory:`` for an in-memory store.
        config : Config or None
            Configuration object; falls back to ``Config()`` defaults.
        """
        self.config = config or Config()
        self.db = Database(db_path)
        self.db.initialize()
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            try:
                from .embeddings import EmbeddingEngine  # noqa: PLC0415
                self._engine = EmbeddingEngine()
            except Exception:
                pass
        return self._engine

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        content: str,
        node_type: str = "factual",
        coreness: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Add a node.

        Factual nodes automatically compute an embedding via the embedding
        engine.  Procedural nodes store ``None`` as embedding.

        Parameters
        ----------
        content : str
            The node content.  Must not be empty.
        node_type : str
            ``"factual"`` or ``"procedural"``.
        coreness : float
            Initial coreness value.
        tags : list[str] or None
            Optional tags attached to the node.

        Returns
        -------
        str
            UUID v4 string identifying the newly created node.

        Raises
        ------
        ValueError
            If *content* is empty.
        """
        node_id = self.db.insert_node(
            content=content, node_type=node_type, coreness=coreness, tags=tags,
        )

        if self.engine is not None and node_type == "factual":
            vector = self.engine.encode(content)
            blob = np.array(vector, dtype=np.float32).tobytes()
            self.db.update_node_embedding(node_id, blob)

        return node_id

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID.

        Parameters
        ----------
        node_id : str
            UUID of the node.

        Returns
        -------
        Node or None
        """
        return self.db.get_node(node_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        return self.db.get_edge(source_id, target_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        forward_strength: float = 0.5,
        backward_strength: float = 0.5,
        relation_type: str = "generic",
        inhibition: float = 0.0,
    ):
        """Create a directed edge between two nodes.

        Parameters
        ----------
        source_id : str
            Source node UUID.
        target_id : str
            Target node UUID.
        forward_strength : float
            Strength of the forward association.
        backward_strength : float
            Strength of the backward association.
        relation_type : str
            One of the allowed relation types (checked by DB constraints).
        inhibition : float
            Inhibition weight.
        """
        self.db.insert_edge(
            source_id=source_id,
            target_id=target_id,
            forward_strength=forward_strength,
            backward_strength=backward_strength,
            relation_type=relation_type,
            inhibition=inhibition,
        )

    # ------------------------------------------------------------------
    # Recall (semantic retrieval)
    # ------------------------------------------------------------------

    def recall(
        self,
        query_text: str,
        top_k: int = 10,
        tags: Optional[list[str]] = None,
    ) -> list[tuple[Node, float]]:
        """Semantic retrieval.

        Scores every non-deleted node by a weighted combination of coreness,
        embedding similarity, and average outgoing edge strength, then returns
        the *top_k* results sorted descending.

        Parameters
        ----------
        query_text : str
            The search query.
        top_k : int
            Maximum number of results to return.
        tags : list[str] or None
            If given, only nodes whose tags intersect with this list are
            considered.

        Returns
        -------
        list[tuple[Node, float]]
            ``(Node, load_score)`` tuples, sorted by *load_score* descending.
        """
        weights = self.config.memory.load_weights
        beta = weights.beta
        gamma = weights.gamma
        delta = weights.delta

        # 1. Encode the query (if engine is available).
        query_vec: Optional[list[float]] = None
        if self.engine is not None:
            query_vec = self.engine.encode(query_text)

        # 2. Gather all non-deleted nodes.
        nodes = self.db.list_nodes(include_deleted=False)

        scored: list[tuple[Node, float]] = []
        for node in nodes:
            # 4. Tag filter: intersection must be non-empty.
            if tags is not None:
                node_tags = set(node.tags)
                if not node_tags.intersection(tags):
                    continue

            # 3a. Relevance (cosine similarity).
            relevance = 0.0
            if query_vec is not None and node.embedding is not None:
                node_vec = np.frombuffer(node.embedding, dtype=np.float32).tolist()
                relevance = self.engine.similarity(query_vec, node_vec)

            # 3b. Average outgoing edge strength.
            edges = self.db.list_edges(source=node.id)
            avg_strength = 0.0
            if edges:
                avg_strength = sum(e.forward_strength for e in edges) / len(edges)

            # 3. Weighted load score.
            load_score = beta * node.coreness + gamma * relevance + delta * avg_strength
            scored.append((node, load_score))

        # 5. Sort descending and trim to top_k.
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    def reinforce_path(
        self,
        node_ids: list[str],
        edge_ids: list[tuple[str, str]],
        success: bool = True,
    ):
        """Strengthen a reasoning path.

        - Each node's coreness receives a decaying boost (factor starts at 1.0
          and is multiplied by 0.8 per node).
        - ``last_accessed`` and ``access_count`` are updated for each node.
        - Each edge's ``base_stability`` is increased by 0.02 (capped at 0.95),
          and ``last_review`` / ``review_count`` are updated.

        Parameters
        ----------
        node_ids : list[str]
            Ordered list of node UUIDs in the path.
        edge_ids : list[tuple[str, str]]
            Ordered list of ``(source_id, target_id)`` pairs.
        success : bool
            Unused in GREEN phase (reserved for future use).
        """
        factor = 1.0
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node is None:
                continue
            new_coreness = node.coreness + 0.01 * factor
            self.db.update_node_coreness(nid, new_coreness)
            self.db.update_node_access(nid)
            factor *= 0.8

        for src, tgt in edge_ids:
            edge = self.db.get_edge(src, tgt)
            if edge is None:
                continue
            new_stability = min(edge.base_stability + 0.02, 0.95)
            self.db.update_edge_stability(src, tgt, new_stability)

    # ------------------------------------------------------------------
    # Error correction
    # ------------------------------------------------------------------

    def correct_error(
        self,
        location: str,
        correction_type: str,
        value: float = 0.0,
        correct_target_id: Optional[str] = None,
        auto_approve: bool = True,
    ):
        """
        错误修正入口。location 格式："edge:src_id->tgt_id" 或 "node:node_id"。

        5 种修正类型：
        - "reduce": 降低边的 base_stability，降幅 = value（上限 0.2）。新值 = max(0, old - value)
        - "add_correction_edge": 从错误边的 source 指向 correct_target_id 创建新边，
           relation_type='correction', forward_strength=0.7, backward_strength=0.3
        - "inhibit": 设置边的 inhibition = value
        - "mark_dubious": 设置边的 is_dubious = True
        - "delete": 软删除节点（is_deleted = True）

        权限：auto_approve=False 时抛 PermissionError("Manual approval required")
        """
        if not auto_approve:
            raise PermissionError("Manual approval required")

        kind, _, rest = location.partition(":")

        if kind == "edge":
            src_tgt = rest.split("->")
            if len(src_tgt) != 2:
                raise ValueError(f"Invalid edge location format: {rest!r}")
            src_id, tgt_id = src_tgt

            if correction_type == "reduce":
                edge = self.db.get_edge(src_id, tgt_id)
                if edge is None:
                    raise ValueError(f"Edge not found: {src_id} -> {tgt_id}")
                reduction = min(value, 0.2)
                new_val = max(0.0, edge.base_stability - reduction)
                self.db.update_edge_stability(src_id, tgt_id, new_val)

            elif correction_type == "add_correction_edge":
                if correct_target_id is None:
                    raise ValueError("correct_target_id is required for add_correction_edge")
                self.db.insert_edge(
                    source_id=src_id,
                    target_id=correct_target_id,
                    relation_type="correction",
                    forward_strength=0.7,
                    backward_strength=0.3,
                )

            elif correction_type == "inhibit":
                self.db._conn.execute(
                    "UPDATE edges SET inhibition = ? WHERE source_id = ? AND target_id = ?",
                    (value, src_id, tgt_id),
                )
                self.db._conn.commit()

            elif correction_type == "mark_dubious":
                self.db._conn.execute(
                    "UPDATE edges SET is_dubious = 1 WHERE source_id = ? AND target_id = ?",
                    (src_id, tgt_id),
                )
                self.db._conn.commit()

            else:
                raise ValueError(f"Unknown correction_type for edge: {correction_type!r}")

        elif kind == "node":
            node_id = rest

            if correction_type == "delete":
                self.db.soft_delete_node(node_id)
            else:
                raise ValueError(f"Unknown correction_type for node: {correction_type!r}")

        else:
            raise ValueError(f"Unknown location kind: {kind!r}")

    # ------------------------------------------------------------------
    # Chain-of-thought logging
    # ------------------------------------------------------------------

    def log_chain(
        self,
        node_ids: list[str],
        edge_ids: list[tuple[str, str]],
        query: str = "",
        success: bool = True,
    ):
        """
        记录思维链到 chain_log 表。
        chain_json 格式：{"nodes": node_ids, "edges": edge_ids, "query": query}
        success: 1 或 0
        """
        chain_data = {"nodes": node_ids, "edges": edge_ids, "query": query}
        chain_json = json.dumps(chain_data, ensure_ascii=False)
        self.db.insert_chain_log(
            chain_json=chain_json,
            success=1 if success else 0,
            error_location=None,
        )

    def get_chains(self, limit: int = 10, success_only: Optional[bool] = None) -> list[dict]:
        """
        获取最近的思维链日志。
        返回 list[dict]，每个 dict 包含 id, timestamp, chain_json, success, error_location。
        如果 success_only=True，只返回成功的链。
        如果 success_only=False，只返回失败的链。
        如果 success_only=None，返回所有。
        """
        chains = self.db.get_chain_logs(limit=limit, success_only=success_only)
        for c in chains:
            c["success"] = bool(c["success"])
        return chains
