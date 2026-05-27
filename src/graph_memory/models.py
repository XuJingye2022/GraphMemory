from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    id: str
    content: str
    coreness: float = 0.5
    node_type: str = "factual"
    tags: list[str] = field(default_factory=list)
    doc_ref: Optional[str] = None
    embedding: Optional[bytes] = None
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    is_deleted: bool = False


@dataclass
class Edge:
    source_id: str
    target_id: str
    forward_strength: float = 0.5
    backward_strength: float = 0.5
    base_stability: float = 0.5
    last_review: str = ""
    review_count: int = 0
    relation_type: str = "generic"
    inhibition: float = 0.0
    is_dubious: bool = False
    created_at: str = ""
