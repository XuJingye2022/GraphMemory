# GraphMemory — AI 图结构长期记忆系统

一个支持**有向联想、动态遗忘、强化学习与错误修正**的图结构记忆系统，为 AI Agent 提供长期记忆能力。

## 核心特性

- **有向联想网络** — 节点间正向/反向强度独立，模拟人类不对称联想
- **动态遗忘曲线** — 不用的记忆随时间衰减，反复使用的路径根深蒂固
- **语义检索** — 基于向量嵌入 + 核心度 + 边强度的加权召回（load score）
- **强化学习** — 成功路径自动增强，失败路径自动削弱
- **错误修正闭环** — 5 种修正方式（降低强度/纠错边/抑制边/标记待验证/删除），三级权限控制
- **身份感知重放** — 按 Agent 角色自动巩固核心职责相关路径
- **定期维护** — 合并相似事实节点、剪枝孤立死节点、清理过期日志
- **独立监控器** — 只读监听链日志，自动生成修正建议
- **REST API** — 15 个端点，APScheduler 自动维护调度

## 快速开始

```bash
# 安装
pip install -e .

# 启动 API 服务
uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload

# 打开文档
open http://localhost:8000/docs
```

## 使用示例

```python
from graph_memory import GraphMemory

mem = GraphMemory("memory.db")

# 存入知识
n1 = mem.add_node("Python uses indentation for code blocks", 
                  node_type="factual", coreness=0.7, tags=["python"])
n2 = mem.add_node("调用 list.append() 原地修改列表",
                  node_type="factual", tags=["python", "list"])
mem.add_edge(n1, n2, relation_type="causes")

# 语义检索
results = mem.recall("python syntax", top_k=5)
for node, score in results:
    print(f"[{score:.2f}] {node.content}")

# 强化成功路径
mem.reinforce_path(node_ids=[n1, n2], edge_ids=[(n1, n2)], success=True)

# 修正错误
mem.correct_error(location=f"edge:{n1}->{n2}", correction_type="inhibit", value=0.8)

# 记录思维链
mem.log_chain(node_ids=[n1, n2], edge_ids=[(n1, n2)], query="How does Python handle lists?")
```

## 项目结构

```
├── src/
│   ├── graph_memory/
│   │   ├── models.py        # Node, Edge 数据模型
│   │   ├── db.py            # SQLite 持久化层（WAL 模式，5 表）
│   │   ├── embeddings.py    # 本地向量嵌入（sentence-transformers）
│   │   ├── algorithms.py    # 衰减曲线、加载分数纯函数
│   │   ├── core.py          # GraphMemory 主引擎
│   │   ├── maintenance.py   # 合并相似、剪枝死节点、重放
│   │   ├── monitor.py       # 独立纠错监控器
│   │   └── config.py        # 配置管理（config.yaml）
│   └── api/
│       └── server.py        # FastAPI REST API（15 端点）
├── tests/                   # 108 个测试
├── config.yaml              # 系统配置
├── agent_profile.yaml       # Agent 身份定义
└── CONTEXT.md               # 领域术语表
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/nodes` | 创建节点 |
| GET | `/nodes/{id}` | 获取节点 |
| DELETE | `/nodes/{id}` | 软删除节点 |
| POST | `/edges` | 创建联想边 |
| GET | `/edges/{source}/{target}` | 获取边 |
| DELETE | `/edges/{source}/{target}` | 删除边 |
| POST | `/recall` | 语义检索 |
| POST | `/reinforce` | 强化路径 |
| POST | `/correct` | 错误修正 |
| POST | `/chains` | 记录思维链 |
| GET | `/chains` | 获取思维链 |
| POST | `/maintenance/merge` | 合并相似节点 |
| POST | `/maintenance/prune` | 剪枝死节点 |
| POST | `/maintenance/replay` | 记忆重放 |
| GET | `/maintenance/status` | 维护状态 |
| GET | `/stats` | 统计信息 |

## 开发

```bash
# 安装开发依赖
pip install -e .

# 运行测试
python -m pytest tests/ -v

# 集成测试（需 DeepSeek API key）
python -m pytest tests/test_agent_e2e.py -v
```

## 设计哲学

记忆不是静态数据库，而是一张**有方向的联想网**。关键洞察：

- **方向不对称** — 从 A 想到 B 容易 ≠ 从 B 想到 A 容易
- **用进废退** — 反复使用的路径增强，不用的自然衰减
- **情绪加速** — 成功/失败对记忆有额外影响
- **先忘记，后巩固** — 新记忆早期衰减快，反复使用后衰减变慢

详见 `docs/AI记忆系统方案构想.md` 和 `docs/PRD.md`。

## License

MIT
