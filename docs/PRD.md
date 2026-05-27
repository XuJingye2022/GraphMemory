# PRD: AI 图结构长期记忆系统

> 基于《AI记忆系统方案构想(v2.0)》及 13 轮设计决策访谈生成。

## Problem Statement

现有 AI Agent 缺乏长期记忆能力：每次对话从零开始，无法记住过去的成功经验、纠正过的错误、或用户的具体偏好。传统知识图谱是静态数据库，不支持"用进废退"的遗忘、情绪驱动的强化、以及错误修正闭环。开发者需要一个**活生生的记忆系统**，让 Agent 像人一样：记住成功流程、淡忘不用的知识、修正错误认知、并按职责优先巩固核心技能。

## Solution

一个基于 **有向联想图** 的 AI Agent 长期记忆系统。核心特征：

- **方向不对称联想**：从 A 想到 B 与从 B 想回 A 是两条独立的强度路径
- **动态遗忘曲线**：不用的边和节点随时间衰减，反复使用的路径根深蒂固
- **强化学习**：成功任务增强路径上的节点和边，失败任务削弱
- **错误修正闭环**：自动/辅助/人工三级纠错，支持抑制边阻断危险联想
- **身份感知重放**：按 Agent 角色自动巩固核心职责相关的记忆路径
- **定期维护**：合并相似事实、剪枝无用节点、清理过期日志

## User Stories

1. As an AI Agent, I want to store a knowledge point as a node, so that I can recall it later when relevant.
2. As an AI Agent, I want to create directed edges between nodes with relation types (sequence, causes, alternative, etc.), so that I can model how knowledge connects.
3. As an AI Agent, I want to recall relevant nodes given a query, ranked by coreness + task relevance + edge strength, so that I can retrieve the most useful information first.
4. As an AI Agent, I want successfully traversed paths to be reinforced (edges strengthened, coreness increased), so that "熟能生巧".
5. As an AI Agent, I want unused edges and nodes to decay over time, so that forgotten knowledge fades naturally.
6. As an AI Agent, I want to create inhibition edges that block dangerous or forbidden associations, so that safety rules cannot be bypassed.
7. As an AI Agent, I want to log my reasoning chains (nodes visited + edges traversed), so that errors can be traced back to their source step.
8. As an AI Agent, I want the system to detect similar factual nodes and merge them, so that redundancy doesn't pollute the graph.
9. As an AI Agent, I want isolated low-coreness nodes to be pruned, so that the graph stays efficient.
10. As an AI Agent, I want to replay core task paths during idle time to prevent skill decay, so that my core competencies remain sharp.
11. As a developer, I want to configure decay rates, merge thresholds, and maintenance intervals via a YAML config file, so that I can tune the system to different use cases.
12. As a developer, I want a FastAPI HTTP API to add nodes/edges, query the graph, trigger maintenance, and view stats, so that I can integrate the memory system into any Agent runtime.
13. As a developer, I want operation logs recording every modification with before/after values and reasons, so that I can audit and roll back changes.
14. As a developer, I want the memory system to be importable as a Python library (`import graph_memory`), so that I can embed it directly into Agent processes.
15. As an AI Agent, I want to distinguish factual knowledge (mergeable) from procedural steps (non-mergeable), so that similar facts get consolidated but alternative workflow steps remain distinct.
16. As an AI Agent, I want tags on nodes to be free-form strings, so that I can develop my own tagging vocabulary without being constrained by a preset enum.

## Implementation Decisions

### Modules to Build

| Module | Responsibility |
|--------|---------------|
| **Core Data Layer** (`db`, `models`) | SQLite schema, node/edge/chain/operation-log tables, WAL mode, migration support |
| **Core Engine** (`core`) | GraphMemory main class: add_node, add_edge, recall, reinforce_path, correct_error |
| **Algorithms** (`algorithms`) | Current strength (decay), load score, reinforcement, forgetting curve |
| **Embeddings** (`embeddings`) | Local text2vec-base-chinese model for semantic similarity, task_relevance scoring |
| **Maintenance** (`maintenance`) | Merge similar factual nodes, prune dead nodes, replay core paths, clean chain logs |
| **Monitor** (`monitor`) | Independent read-only component: detect errors from chain logs, generate correction proposals |
| **API Server** (`api/server`) | FastAPI REST endpoints: CRUD for nodes/edges, recall, reinforce, maintenance triggers, stats |
| **Config** (`config`) | Load config.yaml, manage runtime settings, provide defaults |

### Architecture Decisions

- **Node ID**: UUID v4, auto-generated. Not content-hash based (content may evolve).
- **Node types**: Factual (mergeable) and Procedural (non-mergeable). Distinguished by a `node_type` field.
- **Edge relation_type**: Strict enum — `sequence`, `alternative`, `causes`, `contains`, `similar_to`, `correction`, `generic`. No free-form values.
- **Tags**: Free-form string arrays. No preset enum. Agent develops its own vocabulary.
- **Embeddings**: Local `text2vec-base-chinese` via sentence-transformers. No API calls for embedding.
- **Maintenance scheduling**: APScheduler background thread in FastAPI + manual trigger API endpoints.
- **Chain log**: Stored as `{"nodes": [...], "edges": [...], "query": "..."}` JSON. Retained 90 days then deleted by time.
- **Inhibition**: A `REAL` field on the edge row (0 = no inhibition, > 0 = inhibition strength). Threshold 0.7 blocks activation entirely.
- **Storage**: Single SQLite file with WAL mode. Incremental daily backups + full weekly dump.
- **Correction levels**: Auto (user explicitly identifies node/edge) → Assisted (system proposes, user confirms) → Manual (user-only for safety-critical changes).

### Schema Contract

Nodes table: `id (TEXT PK)`, `content (TEXT)`, `coreness (REAL [0,1])`, `node_type (TEXT: factual|procedural)`, `tags (TEXT/JSON)`, `doc_ref (TEXT nullable)`, `embedding (BLOB)`, timestamps, `access_count`, `is_deleted`.

Edges table: `source_id + target_id (composite PK)`, `forward_strength (REAL)`, `backward_strength (REAL)`, `base_stability (REAL)`, `relation_type (TEXT enum)`, `inhibition (REAL [0,1])`, `is_dubious (BOOL)`, timestamps, `review_count`.

### API Contract

```
POST   /nodes                    — add node
GET    /nodes/{id}               — get node
DELETE /nodes/{id}               — soft-delete node
POST   /edges                    — add edge
GET    /edges/{source}/{target}  — get edge
DELETE /edges/{source}/{target}  — remove edge
POST   /recall                   — recall nodes from query (text + optional tags)
POST   /reinforce                — reinforce a path (node_ids + edge_ids, success flag)
POST   /correct                  — correct an error (target, correction_type, value)
POST   /maintenance/merge        — trigger merge
POST   /maintenance/prune        — trigger prune
POST   /maintenance/replay       — trigger replay
GET    /maintenance/status       — last run times
GET    /stats                    — node count, edge count, avg query latency
GET    /logs                     — operation log timeline
```

## Testing Decisions

- **What to test**: External behavior only — API responses, recall ranking quality, decay behavior, reinforcement effects, merge correctness, prune safety.
- **Not to test**: Internal implementation details of algorithms, SQL query strings, thread scheduling timing.
- **Integration test**: A dedicated script uses deepseek-v4-flash as a simulated Agent. The Agent performs data investigation tasks, writes findings to memory, later retrieves them for similar tasks, and the test verifies recall quality, reinforcement, and correction behavior.
- **API key storage**: `test_config.local.json` (gitignored), containing `api_key` and `base_url`.
- **Unit tests**: Test `GraphMemory` core (add/recall/reinforce/correct), algorithms (decay curve math, load score weighting), and maintenance (merge/prune conditions).

## Out of Scope

- Web visualization UI with force-directed graph (deferred to future iteration)
- Cross-Agent shared memory (`agent_id` field planned but not implemented)
- Parameter auto-tuning (alpha, half_life optimization based on historical success rates)
- External document sync monitoring (doc_ref change detection)
- GPU acceleration for embeddings

## Further Notes

- Initial Agent profile: "数据调查助理" — investigates data trends, writes SQL, searches the web, generates reports. Core tasks include connecting to data sources, writing queries, analyzing results.
- The design document (`docs/AI记忆系统方案构想.md`) is the authoritative concept spec. This PRD is the implementation spec derived from it.
- All 13 design decisions from the grilling session are recorded in `CONTEXT.md` as domain glossary terms.
