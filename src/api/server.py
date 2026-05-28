"""FastAPI server wrapping GraphMemory as a REST API.

All 15 endpoints map to ``GraphMemory`` methods or direct database operations.
APScheduler runs maintenance tasks (merge, prune, replay) on configurable intervals.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from graph_memory import GraphMemory
from graph_memory.maintenance import merge_similar_nodes, prune_dead_nodes, replay_core_paths
from graph_memory.models import Edge, Node

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _node_to_dict(node: Node) -> dict[str, Any]:
    """Convert a Node dataclass to a JSON-serialisable dict.

    The binary *embedding* field is included as a list of floats when present
    so that clients can inspect or use it.
    """
    result: dict[str, Any] = {
        "id": node.id,
        "content": node.content,
        "coreness": node.coreness,
        "node_type": node.node_type,
        "tags": node.tags,
        "doc_ref": node.doc_ref,
        "created_at": node.created_at,
        "last_accessed": node.last_accessed,
        "access_count": node.access_count,
        "is_deleted": node.is_deleted,
    }
    if node.embedding is not None:
        result["embedding"] = list(np.frombuffer(node.embedding, dtype=np.float32))
    return result


def _edge_to_dict(edge: Edge) -> dict[str, Any]:
    """Convert an Edge dataclass to a JSON-serialisable dict."""
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "forward_strength": edge.forward_strength,
        "backward_strength": edge.backward_strength,
        "base_stability": edge.base_stability,
        "last_review": edge.last_review,
        "review_count": edge.review_count,
        "relation_type": edge.relation_type,
        "inhibition": edge.inhibition,
        "is_dubious": edge.is_dubious,
        "created_at": edge.created_at,
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_AGENT_PROFILE_PATH = Path(__file__).resolve().parent.parent.parent / "agent_profile.yaml"


def _load_agent_profile() -> dict:
    if _AGENT_PROFILE_PATH.exists():
        with open(_AGENT_PROFILE_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create GraphMemory instance + start APScheduler for maintenance."""
    memory = GraphMemory(":memory:")
    app.state.memory = memory
    app.state.maintenance_status = {
        "last_merge": None,
        "last_prune": None,
        "last_replay": None,
    }

    scheduler = BackgroundScheduler()
    agent_profile = _load_agent_profile()

    scheduler.add_job(
        lambda: _run_merge(memory, app),
        "interval",
        hours=24,
        id="merge",
    )
    scheduler.add_job(
        lambda: _run_prune(memory, app),
        "interval",
        hours=24,
        id="prune",
    )
    scheduler.add_job(
        lambda: _run_replay(memory, app, agent_profile),
        "interval",
        days=7,
        id="replay",
    )
    scheduler.start()
    app.state.scheduler = scheduler

    yield

    scheduler.shutdown(wait=False)


def _run_merge(memory, app):
    result = merge_similar_nodes(memory)
    app.state.maintenance_status["last_merge"] = datetime.now(timezone.utc).isoformat()


def _run_prune(memory, app):
    result = prune_dead_nodes(memory)
    app.state.maintenance_status["last_prune"] = datetime.now(timezone.utc).isoformat()


def _run_replay(memory, app, agent_profile):
    if agent_profile:
        result = replay_core_paths(memory, agent_profile)
    app.state.maintenance_status["last_replay"] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Graph Memory API", version="0.1.0", lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@app.post("/nodes", status_code=201)
async def add_node(request: Request, body: dict = Body(...)):
    """Create a new memory node."""
    memory: GraphMemory = request.app.state.memory
    node_id = memory.add_node(
        content=body.get("content", ""),
        node_type=body.get("node_type", "factual"),
        coreness=body.get("coreness", 0.5),
        tags=body.get("tags"),
    )
    return {"node_id": node_id}


@app.get("/nodes/{node_id}")
async def get_node(request: Request, node_id: str):
    """Retrieve a node by its ID.  Returns 404 if not found."""
    memory: GraphMemory = request.app.state.memory
    node = memory.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return _node_to_dict(node)


@app.delete("/nodes/{node_id}")
async def delete_node(request: Request, node_id: str):
    """Soft-delete a node via correct_error."""
    memory: GraphMemory = request.app.state.memory
    memory.correct_error(location=f"node:{node_id}", correction_type="delete")
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@app.post("/edges")
async def add_edge(request: Request, body: dict = Body(...)):
    """Create a directed edge between two nodes."""
    memory: GraphMemory = request.app.state.memory
    memory.add_edge(
        source_id=body["source_id"],
        target_id=body["target_id"],
        forward_strength=body.get("forward_strength", 0.5),
        backward_strength=body.get("backward_strength", 0.5),
        relation_type=body.get("relation_type", "generic"),
        inhibition=body.get("inhibition", 0.0),
    )
    return {"status": "ok"}


@app.get("/edges/{source}/{target}")
async def get_edge(request: Request, source: str, target: str):
    """Retrieve an edge by source+target.  Returns 404 if not found."""
    memory: GraphMemory = request.app.state.memory
    edge = memory.db.get_edge(source, target)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    return _edge_to_dict(edge)


@app.delete("/edges/{source}/{target}")
async def delete_edge(request: Request, source: str, target: str):
    """Delete an edge by source+target."""
    memory: GraphMemory = request.app.state.memory
    memory.db._conn.execute(
        "DELETE FROM edges WHERE source_id = ? AND target_id = ?",
        (source, target),
    )
    memory.db._conn.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Recall (semantic retrieval)
# ---------------------------------------------------------------------------


@app.post("/recall")
async def recall(request: Request, body: dict = Body(...)):
    """Semantic retrieval over stored nodes."""
    memory: GraphMemory = request.app.state.memory
    results = memory.recall(
        query_text=body.get("query_text", ""),
        top_k=body.get("top_k", 10),
        tags=body.get("tags"),
    )
    return [
        {"node": _node_to_dict(node), "score": float(score)}
        for node, score in results
    ]


# ---------------------------------------------------------------------------
# Reinforce
# ---------------------------------------------------------------------------


@app.post("/reinforce")
async def reinforce(request: Request, body: dict = Body(...)):
    """Strengthen a reasoning path."""
    memory: GraphMemory = request.app.state.memory
    memory.reinforce_path(
        node_ids=body["node_ids"],
        edge_ids=[tuple(e) for e in body["edge_ids"]],
        success=body.get("success", True),
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Correct (error correction)
# ---------------------------------------------------------------------------


@app.post("/correct")
async def correct(request: Request, body: dict = Body(...)):
    """Apply an error correction to a node or edge."""
    memory: GraphMemory = request.app.state.memory
    memory.correct_error(
        location=body["location"],
        correction_type=body["correction_type"],
        value=body.get("value", 0.0),
        correct_target_id=body.get("correct_target_id"),
        auto_approve=body.get("auto_approve", True),
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chains (chain-of-thought logging)
# ---------------------------------------------------------------------------


@app.post("/chains")
async def log_chain(request: Request, body: dict = Body(...)):
    """Record a reasoning chain."""
    memory: GraphMemory = request.app.state.memory
    memory.log_chain(
        node_ids=body["node_ids"],
        edge_ids=[tuple(e) for e in body["edge_ids"]],
        query=body.get("query", ""),
        success=body.get("success", True),
    )
    return {"status": "ok"}


@app.get("/chains")
async def get_chains(
    request: Request,
    limit: int = 10,
    success_only: Optional[bool] = None,
):
    """Retrieve recent reasoning chain logs."""
    memory: GraphMemory = request.app.state.memory
    return memory.get_chains(limit=limit, success_only=success_only)


# ---------------------------------------------------------------------------
# Maintenance (stubs)
# ---------------------------------------------------------------------------


@app.post("/maintenance/merge")
async def maintenance_merge(request: Request):
    """Trigger a merge pass for similar factual nodes."""
    memory: GraphMemory = request.app.state.memory
    result = merge_similar_nodes(memory)
    request.app.state.maintenance_status["last_merge"] = datetime.now(timezone.utc).isoformat()
    return result


@app.post("/maintenance/prune")
async def maintenance_prune(request: Request):
    """Trigger a pruning pass for dead nodes."""
    memory: GraphMemory = request.app.state.memory
    result = prune_dead_nodes(memory)
    request.app.state.maintenance_status["last_prune"] = datetime.now(timezone.utc).isoformat()
    return result


@app.post("/maintenance/replay")
async def maintenance_replay(request: Request):
    """Trigger a replay pass to strengthen core paths."""
    memory: GraphMemory = request.app.state.memory
    agent_profile = _load_agent_profile()
    result = replay_core_paths(memory, agent_profile) if agent_profile else {"status": "no_profile"}
    request.app.state.maintenance_status["last_replay"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/maintenance/status")
async def maintenance_status(request: Request):
    """Return the last-run timestamps for each maintenance task."""
    return request.app.state.maintenance_status


# ---------------------------------------------------------------------------
# Graph visualization endpoints
# ---------------------------------------------------------------------------


@app.get("/graph")
async def graph(request: Request):
    """Return all nodes and edges for the graph visualization."""
    memory: GraphMemory = request.app.state.memory
    nodes = memory.db.list_nodes(include_deleted=False)
    edges = memory.db.list_edges()
    return {
        "nodes": [_node_to_dict(n) for n in nodes],
        "edges": [
            {
                "source_id": e.source_id,
                "target_id": e.target_id,
                "forward_strength": e.forward_strength,
                "backward_strength": e.backward_strength,
                "relation_type": e.relation_type,
                "inhibition": e.inhibition,
                "is_dubious": e.is_dubious,
            }
            for e in edges
        ],
    }


@app.post("/expand")
async def expand(request: Request, body: dict = Body(...)):
    """Expand from seed nodes out to 2-hop neighbors."""
    memory: GraphMemory = request.app.state.memory
    seed_ids: list[str] = body.get("node_ids", [])
    depth: int = body.get("depth", 2)

    visited = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            out_edges = memory.db.list_edges(source=nid)
            in_edges = memory.db.list_edges(target=nid)
            for e in out_edges + in_edges:
                neighbor = e.target_id if e.source_id == nid else e.source_id
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    expanded_nodes = []
    for nid in visited:
        node = memory.db.get_node(nid)
        if node is not None:
            expanded_nodes.append(_node_to_dict(node))

    expanded_edges = []
    seen_pairs = set()
    for nid in visited:
        out_edges = memory.db.list_edges(source=nid)
        for e in out_edges:
            if e.source_id in visited and e.target_id in visited:
                key = (e.source_id, e.target_id)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    expanded_edges.append({
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "forward_strength": e.forward_strength,
                        "backward_strength": e.backward_strength,
                        "relation_type": e.relation_type,
                        "inhibition": e.inhibition,
                        "is_dubious": e.is_dubious,
                    })

    return {
        "nodes": expanded_nodes,
        "edges": expanded_edges,
        "depth": depth,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@app.get("/stats")
async def stats(request: Request):
    """Return node and edge counts."""
    memory: GraphMemory = request.app.state.memory
    nodes = memory.db.list_nodes(include_deleted=False)
    edges = memory.db.list_edges()
    return {"node_count": len(nodes), "edge_count": len(edges)}
