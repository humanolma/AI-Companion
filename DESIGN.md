# AI 虚拟伴侣 — 项目设计与技术分析

## 一、项目背景

**AI 虚拟伴侣（AI Virtual Companion）** 是一个基于 LangChain + DeepSeek 的 AI 对话应用，核心目标是打造一个有记忆、有情感、有个性的 AI 对话伙伴。

### 核心能力

- 🧠 **记忆系统**：双层记忆架构，短期对话上下文 + ChromaDB 长期记忆，跨会话也能记住用户信息
- 💖 **情感感知**：识别用户情绪（开心/低落/生气/焦虑/平静），动态调整回复语气
- 🎭 **角色人设**：可自定义伴侣的性格、背景故事、说话风格
- ⌨️ **流式输出**：SSE 流式推送，打字机效果实时反馈
- 💾 **对话持久化**：JSON 本地存储，重启不丢失对话记录

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek API (`deepseek-v4-flash`) | OpenAI 兼容接口，成本低 |
| Agent 框架 | LangChain 1.x | Prompt 模板、链式调用、流式输出 |
| 长期记忆 | ChromaDB | 向量数据库，语义检索历史对话 |
| Embedding | 阿里云 DashScope `text-embedding-v3` | 云端 API，无需本地资源 |
| 后端 | FastAPI | REST API + SSE 流式接口 |
| 前端 | 原生 HTML/CSS/JS | 玻璃拟态 UI，零框架依赖 |
| 配置管理 | pydantic-settings | 类型安全的环境变量管理 |

---

## 二、系统设计

### 2.1 分层架构

```
┌──────────────────────────────────────────┐
│  UI 层：FastAPI + 原生 HTML/CSS/JS        │
│  (REST API + SSE 流式接口 + 玻璃拟态UI)   │
├──────────────────────────────────────────┤
│  Agent 层：CompanionAgent 主类            │
│  ├── LLM 封装          (DeepSeek API)     │
│  ├── 情感感知模块      (EmotionDetector)  │
│  ├── 短期记忆管理      (MessagesPlaceholder)│
│  └── 长期记忆管理      (ChromaDB)         │
├──────────────────────────────────────────┤
│  Prompt 层：ChatPromptTemplate            │
├──────────────────────────────────────────┤
│  Config 层：Pydantic Settings             │
└──────────────────────────────────────────┘
```

### 2.2 项目结构

```
ai-companion/
├── main.py                       # 入口文件（uvicorn 启动 FastAPI）
├── requirements.txt
├── src/
│   ├── config/
│   │   └── settings.py           # Pydantic Settings 配置管理
│   ├── agent/
│   │   ├── llm.py                # LLM 初始化（DeepSeek/OpenAI 兼容）
│   │   ├── companion.py          # 虚拟伴侣 Agent 主类
│   │   ├── memory.py             # 长期记忆（ChromaDB + DashScope Embedding）
│   │   └── emotion.py            # 情感感知模块
│   ├── prompts/
│   │   └── companion_prompt.py   # Prompt 模板
│   └── ui/
│       ├── server.py             # FastAPI 后端（REST + SSE）
│       └── static/
│           ├── index.html        # 前端页面（玻璃拟态设计）
│           └── avatar.jpg        # 伴侣头像
├── test_basic.py                 # 基础对话测试
├── test_long_term_memory.py      # 长期记忆测试
├── test_emotion.py               # 情感感知测试
└── test_stream.py                # 流式输出测试
```

### 2.3 双层级记忆架构（核心亮点）

这是系统最核心的设计，解决 AI 对话中"金鱼记忆"的问题：

```
                 用户输入
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  情感检测       语义检索        获取短期
 (额外LLM调用)   ChromaDB       记忆上下文
     │              │              │
     │        召回最相关k条         │
     │        历史记忆 (k=3)       │
     │              │              │
     └──────────────┼──────────────┘
                    ▼
          组装 System Prompt
     (基础人设 + 情绪调整 + 长期记忆)
                    │
                    ▼
               LLM 生成回复
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  更新短期记忆   存入长期记忆    JSON 持久化
 (滑动窗口)     (ChromaDB)     (本地文件)
```

- **短期记忆**：通过 LangChain `MessagesPlaceholder` 保留最近 N 轮对话上下文（默认 20 条消息），直接注入 Prompt，保证对话连贯性。
- **长期记忆**：基于 ChromaDB 向量数据库，每轮对话自动存入，下次对话时通过语义检索召回最相关的 k 条（默认 3 条）历史记忆，注入 System Prompt，实现**跨会话记忆**。

### 2.4 情感感知闭环

```
用户输入
    │
    ▼
LLM 情绪分析 (额外一次调用)
    │  只返回标签词：happy/sad/angry/anxious/neutral
    ▼
识别 5 种情绪
    │
    ▼
生成情绪调整指令 → 注入 System Prompt
    │  • happy  → "分享快乐，语气轻快活泼"
    │  • sad    → "温柔安慰，多倾听，不急着给建议"
    │  • angry  → "保持冷静同理心，先认同感受"
    │  • anxious→ "安抚鼓励，帮助放松"
    │  • neutral→ "保持一贯温柔风格"
    ▼
LLM 生成匹配语气的回复
    │
    ▼
前端实时展示情绪标签（颜色 + 动画）
```

实现位于 `src/agent/emotion.py`，情绪分析 Prompt 设计为极简格式，只返回标签词，减少 token 消耗。

### 2.5 流式输出管道

```
FastAPI SSE (text/event-stream)
    │  事件类型：emotion → chunk → chunk → ... → done
    ▼
前端 fetch + ReadableStream
    │  逐 chunk 读取 + 逐字追加渲染
    ▼
打字机效果实时展示
    │
    ▼
流结束后统一处理：
  • 更新短期记忆（追加 HumanMessage + AIMessage）
  • 存入 ChromaDB 长期记忆
  • JSON 文件持久化
```

关键设计：流式输出期间**不更新记忆**（因为回复还没完整生成），而是在流结束后**一次性批量更新**所有记忆状态。这保证了数据的完整性和一致性。

### 2.6 Embedding 适配层设计

由于 DeepSeek API 不提供 Embeddings 接口，需要自实现适配层。

**方案对比**：

| 阶段 | 方案 | 优点 | 缺点 |
|------|------|------|------|
| 初始 | sentence-transformers 本地模型 (`all-MiniLM-L6-v2`) | 免费、无需联网 | 首次加载慢、占用内存、CPU 推理慢 |
| 当前 | 阿里云 DashScope `text-embedding-v3` API | 速度快、无本地资源开销 | 需额外 API Key、有网络依赖 |

实现位于 `src/agent/memory.py` 的 `DashScopeEmbeddings` 类：

- 继承 LangChain 的 `Embeddings` 抽象基类，实现 `embed_documents()` 和 `embed_query()` 两个方法
- 底层直接调用 DashScope 原生 REST API
- 区分 `text_type` 参数：存储时使用 `"document"` 类型，检索时使用 `"query"` 类型，获得更精准的语义向量

### 2.7 API 接口设计

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端 HTML 页面 |
| GET | `/api/history` | 获取历史对话 + 当前情绪 |
| POST | `/api/chat` | 发送消息（SSE 流式回复） |
| POST | `/api/reset` | 重置当前对话（不删文件） |
| POST | `/api/clear` | 清除所有数据（对话 + 长期记忆） |
| GET | `/api/info` | 获取伴侣名称和性格信息 |

---

## 三、挑战点与优化措施

### 挑战 1：Embedding 方案选型与演进

**问题**：DeepSeek 不提供 Embedding API，但 ChromaDB 必须依赖向量化才能实现语义检索。

**优化措施**：自实现 `DashScopeEmbeddings` 适配层，完全遵循 LangChain `Embeddings` 接口规范（`memory.py` 第 13-42 行），底层调用阿里云 DashScope 原生 API。同时保留了配置切换能力，未来可以无缝切换到其他 Embedding 服务。

```python
# memory.py 核心设计
class DashScopeEmbeddings(Embeddings):
    def embed_documents(self, texts) -> List[List[float]]:
        return self._call_api(texts, text_type="document")  # 存储用

    def embed_query(self, text) -> List[float]:
        return self._call_api([text], text_type="query")[0]  # 检索用
```

### 挑战 2：情绪检测引入额外延迟

**问题**：每次对话需要额外一次 LLM 调用来做情绪分析，串行调用（分析 → 回复）增加了 1-2 秒延迟。

**优化措施**：

- **简单缓存**：`EmotionDetector` 中缓存上一次的输入和检测结果（`emotion.py` 第 60-61 行），相同输入直接命中缓存，避免重复分析。
- **极简 Prompt**：情绪分析 Prompt 要求只返回单个标签词（"happy/sad/... 不返回其他任何内容"），最小化输出 token 数，加快分析速度。
- **异常降级**：LLM 调用失败时自动降级为 `neutral`，不阻塞对话流程。

### 挑战 3：短期记忆窗口管理

**问题**：对话轮次增多后，历史消息会快速撑爆 LLM 上下文窗口，导致 Token 超限或 API 成本激增。

**优化措施**：通过 `max_short_term_history` 配置（默认 20 条）实现**滑动窗口裁剪**（`companion.py` 第 117-118 行）：

```python
if len(self.history) > settings.max_short_term_history:
    self.history = self.history[-settings.max_short_term_history:]
```

同时，长期记忆（ChromaDB）承担了更久远信息的存储和语义检索职责，短期记忆只负责维持最近几轮的对话连贯性，两者形成互补。

### 挑战 4：流式输出与记忆更新的时序协调

**问题**：流式输出期间回复还没生成完整，不能更新记忆；但流结束后必须确保所有记忆状态一致更新。

**优化措施**：`chat_stream()` 方法（`companion.py` 第 129-160 行）采用了**"流中缓冲 + 流后统一处理"**模式：

```
流式阶段:
  full_response = ""
  for chunk in chain.stream(...):
      full_response += chunk.content
      yield chunk.content       # 实时推送给前端
                                  # 此阶段不更新任何记忆

流结束后（一次性批量处理）:
  history.append(HumanMessage)   # 追加用户消息
  history.append(AIMessage)      # 追加完整回复
  long_term_memory.add_memory()  # 存入 ChromaDB
  _save_history()                # JSON 持久化
```

### 挑战 5：对话持久化方案选择

**问题**：需要在服务重启后恢复对话上下文，但项目没有引入正式数据库。

**优化措施**：采用 **JSON 文件存储**（`companion.py` 第 168-198 行），兼顾简洁性和功能完整性：

- 每次对话结束后立即写盘（`_save_history()`），保证数据不丢失
- 启动时自动加载（`_load_history()`），恢复上次对话状态
- 数据格式简洁：`[{"role": "user", "content": "..."}, {"role": "assistant", ...}]`
- 加载失败时降级为空列表，不影响服务启动

> 💡 **潜在优化方向**：当前方案适合单用户场景。如果需要多用户支持，JSON 文件会面临并发写入和数据隔离问题，届时建议升级到 SQLite 或 PostgreSQL。

### 挑战 6：全局单例 Agent 的并发安全性

**问题**：`server.py` 创建了全局单例 `agent`（`companion.py` 第 24 行），所有请求共享同一个实例，包括同一份 `self.history` 和 `self.current_emotion`。

**现状分析**：对于当前的单用户 Web UI 场景，全局单例是合理的，因为任何时候只有一个用户在使用。但架构上已经预留了扩展能力——`LongTermMemory` 支持 `user_id` 参数，`CompanionAgent` 的模块化设计也便于未来按 session 隔离实例。

---

## 四、系统评估与展望

### 优势

| 维度 | 评价 |
|------|------|
| 架构清晰 | 分层明确，模块职责单一，易于理解和维护 |
| 记忆能力 | 双层记忆设计（短期+长期）是同类项目中的亮点 |
| 情感交互 | 5 种情绪识别 + 动态语气调整，交互体验有温度 |
| 自实现适配 | Embedding 适配层展示了良好的接口设计能力 |
| 零框架前端 | 原生 HTML/CSS/JS，无依赖，加载快 |
| 测试覆盖 | 4 个独立测试脚本覆盖核心功能 |

### 可优化方向

1. **情绪检测延迟**：可将情绪分析从串行改为并行（分析 + 回复同时发起），减少用户等待时间；或考虑小模型本地推理
2. **多用户支持**：引入 session 管理，按 user_id 隔离 Agent 状态
3. **记忆召回精度**：可在 ChromaDB 中增加更多元数据维度（时间衰减、重要性评分等）
4. **对话持久化**：JSON 升级为 SQLite，支持并发和更复杂查询
5. **前端增强**：支持 Markdown 渲染、代码高亮、图片消息等
