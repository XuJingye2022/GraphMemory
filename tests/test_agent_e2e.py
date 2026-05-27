"""End-to-end integration test: deepseek-v4-flash as a simulated Agent
using the GraphMemory system for data investigation tasks.

Uses urllib (stdlib) to call the DeepSeek API — no extra dependencies needed.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest

# Paths
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load API config
CONFIG_PATH = ROOT / "test_config.local.json"
with open(CONFIG_PATH) as f:
    API_CONFIG = json.load(f)

API_KEY = API_CONFIG["deepseek_api_key"]
BASE_URL = API_CONFIG["deepseek_base_url"].rstrip("/")
MODEL = API_CONFIG["test_model"]


# ---------------------------------------------------------------------------
# DeepSeek API helper (stdlib urllib — zero extra deps)
# ---------------------------------------------------------------------------

def chat(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Call DeepSeek chat API and return the assistant's reply."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


# Skip if API key is not set
needs_api_key = pytest.mark.skipif(
    API_KEY == "YOUR_API_KEY_HERE",
    reason="DeepSeek API key not configured in test_config.local.json",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory():
    """GraphMemory with pre-loaded 'data investigation' knowledge."""
    # Avoid triggering embedding engine (slow model load).
    # Patch EmbeddingEngine before importing GraphMemory.
    import graph_memory.core as core_mod
    original_engine = getattr(core_mod, "EmbeddingEngine", None)

    class DummyEngine:
        def encode(self, text):
            return [0.0] * 384

        def similarity(self, a, b):
            # Simple keyword-based "similarity" for testing
            return 0.0

    core_mod.EmbeddingEngine = DummyEngine

    from graph_memory import GraphMemory
    mem = GraphMemory(":memory:")

    # Seed the memory with "data investigation" knowledge
    # Simulate findings from investigating last week's sales data
    n1 = mem.add_node(
        content="上周销售总额为 128,500 元，环比增长 12.3%",
        node_type="factual",
        coreness=0.7,
        tags=["销售", "周报", "数据"],
    )
    n2 = mem.add_node(
        content="上周销量最高的产品是 Pro 款无线耳机，售出 342 件",
        node_type="factual",
        coreness=0.65,
        tags=["销售", "产品", "数据"],
    )
    n3 = mem.add_node(
        content="上周退货率为 3.2%，较前一周下降 0.5 个百分点",
        node_type="factual",
        coreness=0.6,
        tags=["销售", "退货", "数据"],
    )
    n4 = mem.add_node(
        content="连接数据库 sales_db，执行查询 SELECT * FROM weekly_sales WHERE week=22",
        node_type="procedural",
        coreness=0.8,
        tags=["SQL", "流程"],
    )
    n5 = mem.add_node(
        content="分析销售趋势：先按日汇总销售额，再计算日均值，最后对比前一周",
        node_type="procedural",
        coreness=0.7,
        tags=["分析", "流程"],
    )

    # Link steps in the investigation workflow
    mem.add_edge(n4, n5, relation_type="sequence", forward_strength=0.9)
    mem.add_edge(n1, n2, relation_type="contains", forward_strength=0.7)
    mem.add_edge(n1, n3, relation_type="contains", forward_strength=0.7)

    # Log a successful investigation chain
    mem.log_chain(
        node_ids=[n4, n5, n1, n2],
        edge_ids=[(n4, n5), (n5, n1), (n1, n2)],
        query="查询并分析上周销售数据",
        success=True,
    )

    yield mem

    # Restore original
    if original_engine is not None:
        core_mod.EmbeddingEngine = original_engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentMemoryIntegration:
    """Simulate an AI Agent using the memory system for data investigation."""

    @needs_api_key
    def test_agent_recalls_stored_knowledge(self, memory):
        """Agent recalls sales data, feeds it to LLM, gets informed answer."""
        # Step 1: Agent queries memory with a question
        recalled = memory.recall("上周销售数据总结", top_k=3)

        assert len(recalled) > 0, "Recall should return stored knowledge"

        # Step 2: Build context from recalled nodes for the LLM
        context_items = []
        for node, score in recalled:
            context_items.append(f"- [{score:.2f}] {node.content}")

        context = "\n".join(context_items)

        # Step 3: Ask deepseek-v4-flash to answer using the recalled context
        prompt = f"""根据以下记忆系统检索到的信息，请用中文总结上周的销售情况：

{context}

请用一段话概括，保持简洁。"""

        answer = chat(prompt)

        # Step 4: Verify the answer references stored knowledge
        assert "128,500" in answer or "12.3" in answer or "无线耳机" in answer or "342" in answer, (
            f"LLM answer should reference stored data, got: {answer[:200]}"
        )

        # Step 5: Reinforce the successful path
        node_ids = [n.id for n, _ in recalled]
        edge_ids = []
        for i in range(len(node_ids) - 1):
            edge_ids.append((node_ids[i], node_ids[i + 1]))
        if edge_ids:
            memory.reinforce_path(node_ids=node_ids, edge_ids=edge_ids, success=True)

        assert len(answer) > 20, "Answer should be substantive"

    @needs_api_key
    def test_agent_corrects_misremembered_fact(self, memory):
        """Agent stores wrong info, then corrects it via correction edge."""
        wrong_id = memory.add_node(
            content="上周退货率为 5.0%",
            node_type="factual",
            coreness=0.5,
            tags=["销售", "退货"],
        )
        correct_id = memory.add_node(
            content="上周退货率实际为 3.2%",
            node_type="factual",
            coreness=0.85,  # High coreness so it ranks above the error
            tags=["销售", "退货", "修正"],
        )
        memory.add_edge(wrong_id, correct_id, relation_type="correction")

        recalled = memory.recall("退货率", top_k=10)
        contents = [n.content for n, _ in recalled]

        assert any("3.2%" in c for c in contents), (
            f"Correct fact should appear in recall results, got: {contents}"
        )

    @needs_api_key
    def test_agent_investigates_with_llm(self, memory):
        """Full end-to-end: Agent receives a question, recalls memory,
        calls LLM with context, and the LLM produces an answer."""
        question = "上周哪个产品卖得最好？退货率是多少？"

        # Recall relevant nodes
        recalled = memory.recall(question, top_k=5)
        assert len(recalled) > 0

        # Build context
        context = "\n".join(
            f"- {n.content}" for n, _ in recalled
        )

        # Ask LLM with context
        prompt = f"""你是一个数据分析助手。根据以下记忆系统检索到的信息回答问题。

记忆信息：
{context}

问题：{question}

请基于记忆信息直接回答。如果记忆中没有相关信息，请说明。"""

        answer = chat(prompt)
        assert len(answer) > 10, "LLM should produce an answer"

        # Verify key facts are referenced
        has_product = "无线耳机" in answer or "342" in answer or "Pro" in answer
        has_return = "3.2%" in answer or "退货" in answer
        assert has_product or has_return, (
            f"Answer should reference stored facts, got: {answer[:300]}"
        )

    @needs_api_key
    def test_chain_log_persistence(self, memory):
        """Verify chain logs are stored and retrievable."""
        # Log a new chain
        memory.log_chain(
            node_ids=["a", "b", "c"],
            edge_ids=[("a", "b"), ("b", "c")],
            query="测试链日志",
            success=True,
        )

        chains = memory.get_chains(limit=5)
        assert len(chains) >= 1
        assert chains[0]["success"] is True

    def test_memory_persistence_across_sessions(self, tmp_path):
        """Memory written to a file persists after closing and reopening."""
        db_path = str(tmp_path / "test_persist.db")

        import graph_memory.core as core_mod
        original = getattr(core_mod, "EmbeddingEngine", None)
        core_mod.EmbeddingEngine = type("Dummy", (), {
            "encode": lambda s, t: [0.0] * 384,
            "similarity": lambda s, a, b: 0.0,
        })

        from graph_memory import GraphMemory

        # Session 1: write
        mem1 = GraphMemory(db_path)
        nid = mem1.add_node("持久化测试数据", tags=["test"])
        mem1.add_edge(nid, nid, relation_type="generic")

        # Session 2: reopen and read
        mem2 = GraphMemory(db_path)
        node = mem2.get_node(nid)
        assert node is not None
        assert node.content == "持久化测试数据"

        if original:
            core_mod.EmbeddingEngine = original

    def test_stats_reflect_state(self, memory):
        """Stats endpoint reflects current graph state."""
        # Count nodes (we seeded 5 in the fixture)
        nodes = memory.db.list_nodes(include_deleted=False)
        edges = memory.db.list_edges()
        assert len(nodes) >= 5
        assert len(edges) >= 3
