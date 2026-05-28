# GraphMemory 使用文档

## 快速启动

```bash
# 安装
pip install -e .

# 启动服务（嵌入模型首次加载需 2-5 分钟）
PYTHONPATH=src python -m uvicorn api.server:app --host 127.0.0.1 --port 8000

# 打开可视化界面
open http://localhost:8000
```

---

## 一、核心概念

| 概念 | 说明 |
|------|------|
| **节点 (Node)** | 一个知识点。分为事实性（可合并）和流程性（不可合并，独立步骤）。 |
| **边 (Edge)** | 有向联想。正向强度 = A→B 的容易程度，反向强度独立。 |
| **Coreness** | 节点自身的核心度 [0,1]，越高越重要，可视化中越靠近顶部。 |
| **联想类型** | `sequence`(顺序) / `causes`(因果) / `contains`(包含) / `alternative`(替代) / `similar_to`(相似) / `correction`(纠正) / `generic`(通用) |
| **抑制边** | `inhibition > 0.7` 时强制阻断该联想。 |
| **工作记忆区** | 直接通过 `/recall` 检索到的节点，可视化中用红色轮廓标记。 |
| **深度思考区** | 从召回节点扩散 2+ 跳到达的节点，可视化中用蓝色轮廓标记。 |

## 二、API 使用

### 存知识

```python
import requests

# 添加事实性节点
r = requests.post("http://localhost:8000/nodes", json={
    "content": "Python 使用缩进定义代码块",
    "node_type": "factual",
    "coreness": 0.7,
    "tags": ["python", "语法"]
})
node_id = r.json()["node_id"]

# 添加流程性节点
r = requests.post("http://localhost:8000/nodes", json={
    "content": "执行 pip install flask",
    "node_type": "procedural",
    "tags": ["部署", "python"]
})

# 连接两个节点（联想）
requests.post("http://localhost:8000/edges", json={
    "source_id": node_a,
    "target_id": node_b,
    "relation_type": "causes",
    "forward_strength": 0.8
})
```

### 检索知识

```python
# 语义检索 —— 召回最相关的 top-k 个节点
r = requests.post("http://localhost:8000/recall", json={
    "query_text": "Python 语法规则",
    "top_k": 5,
    "tags": ["python"]
})
results = r.json()
for item in results:
    print(f"[{item['score']:.3f}] {item['node']['content']}")
```

### 强化路径（学习）

```python
# 一次成功推理后，强化走过的节点和边
requests.post("http://localhost:8000/reinforce", json={
    "node_ids": [n1, n2, n3],
    "edge_ids": [[n1, n2], [n2, n3]],
    "success": True
})
```

### 错误修正

```python
# 降低错误边的强度（上限 0.2）
requests.post("http://localhost:8000/correct", json={
    "location": "edge:uuid1->uuid2",
    "correction_type": "reduce",
    "value": 0.15
})

# 添加纠正边（从错误节点指向正确节点）
requests.post("http://localhost:8000/correct", json={
    "location": "edge:wrong_id->wrong_target",
    "correction_type": "add_correction_edge",
    "correct_target_id": "correct_node_id"
})

# 抑制危险联想
requests.post("http://localhost:8000/correct", json={
    "location": "edge:uuid1->uuid2",
    "correction_type": "inhibit",
    "value": 0.8
})

# 标记可疑
requests.post("http://localhost:8000/correct", json={
    "location": "edge:uuid1->uuid2",
    "correction_type": "mark_dubious"
})

# 软删除节点
requests.post("http://localhost:8000/correct", json={
    "location": "node:uuid1",
    "correction_type": "delete"
})
```

### 思维链日志

```python
# 记录推理过程
requests.post("http://localhost:8000/chains", json={
    "node_ids": [n1, n2, n3],
    "edge_ids": [[n1, n2], [n2, n3]],
    "query": "用户询问如何安装 Flask",
    "success": True
})

# 查看最近的推理链
requests.get("http://localhost:8000/chains?limit=5&success_only=true")
```

### 维护

```python
# 手动触发合并相似节点
requests.post("http://localhost:8000/maintenance/merge")

# 手动触发剪枝死节点
requests.post("http://localhost:8000/maintenance/prune")

# 手动触发记忆重放（按 role 强化核心路径）
requests.post("http://localhost:8000/maintenance/replay")

# 查看维护状态
requests.get("http://localhost:8000/maintenance/status")

# 统计信息
requests.get("http://localhost:8000/stats")
```

## 三、可视化界面

`http://localhost:8000` 打开后：

**布局规则：**
- 节点垂直位置 = coreness（越靠近顶部越核心）
- 节点大小 = coreness（越大越核心）
- 事实性节点 = 蓝紫色，流程性节点 = 橙棕色

**边颜色含义：**
| 样式 | 含义 |
|------|------|
| 🟢 绿色粗线 | 强联想 (forward_strength ≥ 0.6) |
| ⚫ 灰色细线 | 弱联想 (0.3 ~ 0.6) |
| 🔴 红色虚线 | 抑制边 (inhibition ≥ 0.7，阻断该联想) |
| 🟡 橙色虚线 | 可疑边 (is_dubious) |

**交互：**

| 操作 | 效果 |
|------|------|
| 在搜索框输入文字 → 点"检索" | 匹配节点被**红色外环**标记（工作记忆区） |
| 点"展开" | 从召回节点扩散 2 跳，新增节点被**蓝色外环**标记（深度思考区） |
| 鼠标悬停节点 | 显示 content、coreness、tags 等详情 |
| 鼠标悬停边 | 显示正/反向强度、relation_type |
| 拖拽节点 | 手动调整布局 |
| 滚轮缩放 | 放大/缩小图谱 |
| 点"重置" | 清除检索/展开标记 |

## 四、配置

`config.yaml` 关键参数：

```yaml
memory:
  initial_half_life: 24    # 半衰期（小时），决定遗忘速度
  alpha: 0.2               # 复习增益因子，越高复习一次忘得越慢
  strength_threshold: 0.3  # 低于此强度的边在检索时忽略
  inhibition_threshold: 0.7 # 抑制强度超过此值时阻断联想
  load_weights:
    beta: 0.4    # coreness 权重
    gamma: 0.4   # 语义相关性权重
    delta: 0.2   # 出边平均强度权重
  merge_similarity: 0.9    # 合并节点时的相似度阈值
```

`agent_profile.yaml` — 重放系统使用：

```yaml
agent:
  name: "数据调查助理"
  description: "帮助用户调查和分析数据"
roles:
  - "数据分析"
  - "SQL 查询"
core_tasks:
  - "连接数据源"
  - "编写查询"
  - "生成报告"
```

## 五、Python 库方式使用

```python
from graph_memory import GraphMemory

mem = GraphMemory("memory.db")

# 存
n1 = mem.add_node("Flask 是一个轻量级 Web 框架", tags=["python", "web"])
n2 = mem.add_node("Flask 默认使用 Jinja2 模板引擎", tags=["python", "template"])
mem.add_edge(n1, n2, relation_type="contains", forward_strength=0.8)

# 检索
results = mem.recall("Python web 框架", top_k=5)

# 强化
mem.reinforce_path(node_ids=[n1, n2], edge_ids=[(n1, n2)])

# 修正
mem.correct_error(location=f"edge:{n1}->{n2}", correction_type="reduce", value=0.1)
```

## 六、作为 Claude Code Skill 集成的建议

将记忆系统作为 Claude Code 的 `memory` 工具使用：

1. **持久化路径**：指定固定数据库路径 `graph_memory.py` 或 SQLite 文件
2. **每次回答前**：用 `POST /recall` 检索相关记忆作为上下文
3. **每次回答后**：用 `POST /chains` 记录本次推理路径
4. **收到纠正时**：用 `POST /correct` 修正错误记忆
5. **定期维护**：每天调用一次 `POST /maintenance/prune`

## 七、测试

```bash
# 全量测试
python -m pytest tests/ -v

# 特定模块
python -m pytest tests/test_db.py -v
python -m pytest tests/test_core.py -v

# 集成测试（需 DeepSeek API key）
python -m pytest tests/test_agent_e2e.py -v
```
