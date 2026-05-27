import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from .models import Edge, Node


class Database:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def initialize(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                coreness REAL DEFAULT 0.5,
                node_type TEXT DEFAULT 'factual' CHECK(node_type IN ('factual','procedural')),
                tags TEXT DEFAULT '[]',
                doc_ref TEXT,
                embedding BLOB,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT REFERENCES nodes(id),
                target_id TEXT REFERENCES nodes(id),
                forward_strength REAL DEFAULT 0.5,
                backward_strength REAL DEFAULT 0.5,
                base_stability REAL DEFAULT 0.5,
                last_review TEXT,
                review_count INTEGER DEFAULT 0,
                relation_type TEXT NOT NULL DEFAULT 'generic' CHECK(relation_type IN ('sequence','alternative','causes','contains','similar_to','correction','generic')),
                inhibition REAL DEFAULT 0.0,
                is_dubious INTEGER DEFAULT 0,
                created_at TEXT,
                PRIMARY KEY(source_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS operation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                operator TEXT,
                operation TEXT,
                target_id TEXT,
                old_value TEXT,
                new_value TEXT,
                reason TEXT
            );

            CREATE TABLE IF NOT EXISTS chain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                chain_json TEXT NOT NULL,
                success INTEGER,
                error_location TEXT
            );

            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self._conn.commit()

    def execute(self, sql: str):
        return self._conn.execute(sql)

    def insert_node(
        self,
        content: str,
        node_type: str = "factual",
        coreness: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> str:
        if not content:
            raise ValueError("content must not be empty")

        node_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        tags_json = json.dumps(tags or [])

        self._conn.execute(
            """INSERT INTO nodes (id, content, coreness, node_type, tags, created_at, last_accessed, access_count, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)""",
            (node_id, content, coreness, node_type, tags_json, now, now),
        )
        self._conn.commit()
        return node_id

    def get_node(self, node_id: str) -> Optional[Node]:
        if not node_id:
            return None

        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()

        if row is None:
            return None

        return Node(
            id=row["id"],
            content=row["content"],
            coreness=row["coreness"],
            node_type=row["node_type"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            doc_ref=row["doc_ref"],
            embedding=row["embedding"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            is_deleted=bool(row["is_deleted"]),
        )

    def insert_edge(
        self,
        source_id: str,
        target_id: str,
        forward_strength: float = 0.5,
        backward_strength: float = 0.5,
        relation_type: str = "generic",
        inhibition: float = 0.0,
    ):
        """插入边。source_id 或 target_id 指向不存在的节点时抛 sqlite3.IntegrityError。
        relation_type 不合法时抛 sqlite3.IntegrityError（CHECK 约束）。
        重复 (source_id, target_id) 抛 sqlite3.IntegrityError（PK 冲突）。"""
        now = datetime.now().isoformat()
        self._conn.execute(
            """INSERT INTO edges
               (source_id, target_id, forward_strength, backward_strength,
                base_stability, last_review, review_count, relation_type,
                inhibition, is_dubious, created_at)
               VALUES (?, ?, ?, ?, 0.5, ?, 0, ?, ?, 0, ?)""",
            (
                source_id,
                target_id,
                forward_strength,
                backward_strength,
                now,
                relation_type,
                inhibition,
                now,
            ),
        )
        self._conn.commit()

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """按 (source_id, target_id) 查边。不存在返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
            (source_id, target_id),
        ).fetchone()

        if row is None:
            return None

        return Edge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            forward_strength=row["forward_strength"],
            backward_strength=row["backward_strength"],
            base_stability=row["base_stability"],
            last_review=row["last_review"] or "",
            review_count=row["review_count"],
            relation_type=row["relation_type"],
            inhibition=row["inhibition"],
            is_dubious=bool(row["is_dubious"]),
            created_at=row["created_at"] or "",
        )

    def list_edges(self, source: Optional[str] = None, target: Optional[str] = None) -> list[Edge]:
        """列出边。source 非空则过滤出边，target 非空则过滤入边，都空则返回全部。
        两者同时非空时相当于 get_edge 的列表版。"""
        if source is not None and target is not None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
                (source, target),
            ).fetchall()
        elif source is not None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (source,)
            ).fetchall()
        elif target is not None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ?", (target,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM edges").fetchall()

        result = []
        for row in rows:
            result.append(
                Edge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    forward_strength=row["forward_strength"],
                    backward_strength=row["backward_strength"],
                    base_stability=row["base_stability"],
                    last_review=row["last_review"] or "",
                    review_count=row["review_count"],
                    relation_type=row["relation_type"],
                    inhibition=row["inhibition"],
                    is_dubious=bool(row["is_dubious"]),
                    created_at=row["created_at"] or "",
                )
            )
        return result

    # ------------------------------------------------------------------
    # Additional helpers for GraphMemory
    # ------------------------------------------------------------------

    def update_node_embedding(self, node_id: str, embedding_blob: bytes):
        """Set the embedding BLOB for an existing node."""
        self._conn.execute(
            "UPDATE nodes SET embedding = ? WHERE id = ?",
            (embedding_blob, node_id),
        )
        self._conn.commit()

    def list_nodes(self, include_deleted: bool = False) -> list[Node]:
        """Return all nodes, optionally including soft-deleted ones."""
        if include_deleted:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE is_deleted = 0"
            ).fetchall()

        result = []
        for row in rows:
            result.append(
                Node(
                    id=row["id"],
                    content=row["content"],
                    coreness=row["coreness"],
                    node_type=row["node_type"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    doc_ref=row["doc_ref"],
                    embedding=row["embedding"],
                    created_at=row["created_at"],
                    last_accessed=row["last_accessed"],
                    access_count=row["access_count"],
                    is_deleted=bool(row["is_deleted"]),
                )
            )
        return result

    def update_node_coreness(self, node_id: str, new_coreness: float):
        """Update the coreness value of a node."""
        self._conn.execute(
            "UPDATE nodes SET coreness = ? WHERE id = ?",
            (new_coreness, node_id),
        )
        self._conn.commit()

    def update_node_access(self, node_id: str):
        """Update last_accessed and increment access_count for a node."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE nodes SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (now, node_id),
        )
        self._conn.commit()

    def update_edge_stability(
        self, source_id: str, target_id: str, new_stability: float
    ):
        """Update base_stability, last_review, and increment review_count for an edge."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE edges SET base_stability = ?, last_review = ?, review_count = review_count + 1 WHERE source_id = ? AND target_id = ?",
            (new_stability, now, source_id, target_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chain-of-thought logging
    # ------------------------------------------------------------------

    def insert_chain_log(
        self,
        chain_json: str,
        success: int,
        error_location: Optional[str] = None,
    ) -> int:
        """Insert a chain-of-thought log entry. Returns the auto-incremented id."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO chain_log (timestamp, chain_json, success, error_location) VALUES (?, ?, ?, ?)",
            (now, chain_json, success, error_location),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_chain_logs(
        self,
        limit: int = 10,
        success_only: Optional[bool] = None,
    ) -> list[dict]:
        """Retrieve chain log entries ordered by timestamp DESC."""
        if success_only is True:
            rows = self._conn.execute(
                "SELECT * FROM chain_log WHERE success = 1 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        elif success_only is False:
            rows = self._conn.execute(
                "SELECT * FROM chain_log WHERE success = 0 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM chain_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "chain_json": row["chain_json"],
                    "success": row["success"],
                    "error_location": row["error_location"],
                }
            )
        return result

    # ------------------------------------------------------------------
    # Soft delete
    # ------------------------------------------------------------------

    def soft_delete_node(self, node_id: str):
        """Soft-delete a node by setting is_deleted = 1."""
        self._conn.execute(
            "UPDATE nodes SET is_deleted = 1 WHERE id = ?",
            (node_id,),
        )
        self._conn.commit()
