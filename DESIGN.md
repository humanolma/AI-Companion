# AI 智能助手 — 项目设计文档

> **面向读者**：接手本项目的开发者。本文档覆盖完整项目结构、模块职责、数据流、关键设计决策、调试方法与常见问题，帮助快速定位代码和排查问题。

---

## 一、项目骨架：30 秒速览

```
用户浏览器  →  FastAPI SSE  →  CompanionAgent  →  DeepSeek API
                  ↑                  │
                  │    ┌─────────────┼─────────────┐
                  │    ▼             ▼             ▼
                  │  Emotion      Memory       MCP Tools
                  │ (情绪识别)  (双层记忆)  (高德/Tavily)
                  │    │             │             │
                  │    └─────────────┼─────────────┘
                  │                  ▼
                  │           System Prompt 组装
                  │                  │
                  └────── 流式回复 ←─┘
```

**一句话描述**：用户输入文字 → 情绪检测 → 检索短期/长期记忆 → 组装 Prompt → LLM 生成回复（可选调 MCP 工具）→ 流式推送到浏览器 → 更新记忆。

---

## 二、完整项目结构

```
AI-Companion/
├── main.py                          # [入口] uvicorn 启动 FastAPI
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板（复制为 .env 使用）
├── .gitignore
│
├── BRD.md                           # 业务需求说明书（产品/功能规划）
├── DESIGN.md                        # ← 当前文件：技术设计文档
├── README.md                        # 项目简介
│
├── test_basic.py                    # [测试] 基础对话功能
├── test_emotion.py                  # [测试] 情绪识别链路
├── test_long_term_memory.py         # [测试] 长期记忆存取与检索
├── test_stream.py                   # [测试] 流式输出
│
├── assets/
│   └── companion_avatar.svg         # 头像 SVG，前端引用
│
├── data/                            # 运行时数据（.gitignore）
│   ├── chat_history.json            # 对话历史持久化文件
│   ├── user_profile.json            # 用户画像（自动提取）
│   ├── usage.json                   # 用量统计（每日 Token + 费用）
│   ├── calendar.json                # 日程数据
│   ├── emotion_history.json         # 情绪历史记录
│   ├── personas.json                # 人设预设（5 套）
│   ├── calendar_ics/                # .ics 日历文件
│   ├── mcp_servers.json             # 额外 MCP 服务器配置（可选）
│   ├── chroma/                      # ChromaDB 向量数据库目录
│   └── logs/                        # 日志文件（按天切割，保留 30 天）
│
└── src/
    ├── config/
    │   └── settings.py              # [配置] Pydantic Settings，所有环境变量
    │
    ├── agent/                       # [核心] Agent 层
    │   ├── companion.py             # CompanionAgent 主类：对话编排、记忆管理
    │   ├── llm.py                   # LLM 初始化：ChatOpenAI 封装 DeepSeek
    │   ├── memory.py                # 长期记忆：ChromaDB + DashScope Embedding
    │   ├── emotion.py               # 情感感知：LLM 情绪分析 + 自适应回复
    │   ├── tools.py                 # MCP 工具：高德 + Tavily + 时间
    │   ├── usage.py                 # 用量追踪：Token 估算 + 费用统计
    │   ├── profile.py               # 用户画像：自动提取 + 结构化存储
    │   └── calendar.py              # 日程管理：JSON + .ics 导出 + LangChain Tool
    │
    ├── utils/
    │   └── logger.py                # 统一日志配置（TimedRotatingFileHandler）
    │
    └── ui/                          # [前端] Web 交互层
        ├── server.py                # FastAPI：REST + SSE + 生命周期 + 导出/搜索
        └── static/
            ├── index.html           # 前端页面：玻璃拟态 UI + Markdown + 搜索
            └── avatar.jpg           # 头像图片
```

---

## 三、模块详解

### 3.1 `main.py` — 应用入口

**职责**：启动 uvicorn 服务器，屏蔽第三方库噪音日志。

```python
# 核心逻辑
uvicorn.run("src.ui.server:app", host="0.0.0.0", port=8080)
```

- 设置环境变量屏蔽 huggingface/tokenizers 的进度条和警告
- 端口可通过 `PORT` 环境变量覆盖
- 本身不含业务逻辑，所有能力在 `src/` 下

---

### 3.2 `src/config/settings.py` — 配置中枢

**类**：`Settings(pydantic_settings.BaseSettings)`

**全局单例**：`settings = Settings()`，模块导入时自动从 `.env` 读取。

| 分类 | 字段 | 默认值 | 说明 |
|------|------|--------|------|
| **API Key** | `deepseek_api_key` | `""` | 必填，核心 LLM |
| | `dashscope_api_key` | `""` | 必填，Embedding 向量化 |
| | `amap_maps_api_key` | `""` | 可选，高德地图工具 |
| | `tavily_api_key` | `""` | 可选，联网搜索 |
| **模型** | `model_name` | `deepseek-v4-flash` | LLM 模型 |
| | `temperature` | `0.7` | 生成温度 |
| **记忆** | `chroma_persist_dir` | `./data/chroma` | ChromaDB 目录 |
| | `memory_retrieval_k` | `3` | 每次检索返回条数 |
| | `max_short_term_history` | `20` | 短期记忆上限（条） |
| **人设** | `companion_name` | `小梦` | 角色名 |
| | `companion_personality` | `温柔、善解人意、有点俏皮` | 性格 |
| | `companion_backstory` | `""` | 背景故事 |
| **MCP** | `mcp_servers_json` | `./data/mcp_servers.json` | 额外 MCP 服务器配置 |
| **持久化** | `chat_history_file` | `./data/chat_history.json` | 对话存储路径 |

> **修改指南**：新增配置项只需在此文件 `Settings` 类中加字段，`.env` 中对应添加即可，全局通过 `settings.xxx` 访问。

---

### 3.3 `src/agent/llm.py` — LLM 初始化

**职责**：创建 LangChain `ChatOpenAI` 实例，封装 DeepSeek API。

```python
def get_llm(temperature: float | None = None, streaming: bool = True):
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature or settings.temperature,
        streaming=streaming,
    )
```

- `streaming=True` 时返回的 LLM 支持 `chain.stream()` 流式调用
- `streaming=False` 时用于情绪分析、工具决策等非流式场景

---

### 3.4 `src/agent/companion.py` — Agent 主类（最核心文件）

**类**：`CompanionAgent`

这是整个系统的**编排器**。所有模块（记忆、情绪、工具、LLM）在此汇聚。

#### 3.4.1 初始化流程

```
CompanionAgent(use_long_term_memory=True, use_emotion=True)
    │
    ├── 1. 创建 get_llm() 实例
    ├── 2. 如果 use_long_term_memory → 初始化 LongTermMemory (ChromaDB)
    ├── 3. 如果 use_emotion → 初始化 EmotionDetector
    ├── 4. _load_history() → 从 JSON 恢复上次对话
    └── 5. self.tools = []  (由外部 server.py 注入)
```

#### 3.4.2 关键方法

| 方法 | 说明 |
|------|------|
| `_build_system_prompt()` | 组装 System Prompt：基础人设 + 情绪调整指令 + 长期记忆 |
| `_build_prompt()` | 组装 LangChain `ChatPromptTemplate`，含 `MessagesPlaceholder` 短期记忆 |
| `_prepare_context()` | 完整上下文：情绪检测 → 记忆检索 → System Prompt 组装 |
| `chat_stream(user_input)` | **主入口**：流式对话生成器，yield 逐字推给前端 |
| `_run_with_tools(response, context)` | 工具调用循环：检测 tool_calls → 执行 → 回填结果 → 重新生成（最多 5 轮） |
| `set_tools(tools)` | 注入 MCP 工具列表（由 server.py 调用） |
| `_save_history()` / `_load_history()` | JSON 读写对话历史 |
| `reset()` | 清空短期记忆（不删文件、不删长期记忆） |
| `clear_all_data()` | 删除所有数据（对话 + ChromaDB + JSON） |

#### 3.4.3 对话完整链路

```
用户输入 "今天天气怎么样"
    │
    ▼
chat_stream(user_input)
    │
    ├── 1. _prepare_context(user_input)
    │       ├── emotion.detect(user_input)           → "neutral"
    │       ├── long_term_memory.retrieve(user_input) → 召回 k=3 条记忆
    │       └── _build_system_prompt(emotion, memories)
    │           生成完整 System Prompt（含人设 + 情绪指令 + 记忆）
    │
    ├── 2. chain.stream(context)  →  开始流式生成
    │
    ├── 3. 检测 response.tool_calls  →  if 有工具调用:
    │       └── _run_with_tools(response, context)
    │             ├── 执行工具 (maps_weather / tavily_search)
    │             ├── 将结果作为 ToolMessage 追加到上下文
    │             └── 重新调用 LLM → GOTO step 2（最多 5 轮）
    │
    ├── 4. 流式 yield chunk  →  推送到前端
    │
    └── 5. 流结束后（一次性批量）:
            ├── self.history.append(HumanMessage)
            ├── self.history.append(AIMessage)
            ├── self.long_term_memory.add_memory(user_input, full_response)
            └── _save_history()
```

#### 3.4.4 工具调用循环（tool calling loop）

- 最多 5 轮迭代，防止死循环
- 每一轮：LLM 返回 tool_calls → Agent 执行工具 → 结果追加为 ToolMessage → 再次交给 LLM
- 工具执行通过 `_MCPTool._run()` 走 `run_coroutine_threadsafe` 回到原始事件循环，避免跨循环死锁

---

### 3.5 `src/agent/memory.py` — 长期记忆

**类**：
| 类 | 职责 |
|------|------|
| `DashScopeEmbeddings` | 阿里云 DashScope Embedding API 适配层，继承 LangChain `Embeddings` |
| `LongTermMemory` | ChromaDB 管理：存入、检索、清除 |

#### DashScopeEmbeddings 设计要点

```python
class DashScopeEmbeddings(Embeddings):
    def embed_documents(self, texts) -> List[List[float]]:
        return self._call_api(texts, text_type="document")  # 存储时用

    def embed_query(self, text) -> List[float]:
        return self._call_api([text], text_type="query")[0]  # 检索时用
```

- `text_type="document"` 用于存储记忆时向量化
- `text_type="query"` 用于检索时向量化，DashScope 对两种类型有不同优化
- 底层直接调用 DashScope REST API（`POST /compatible-mode/v1/embeddings`）

#### LongTermMemory 关键方法

```python
class LongTermMemory:
    def add_memory(user_input, ai_response):
        """存入：user_input + ai_response 拼接为一条记忆，向量化后写入 ChromaDB"""

    def retrieve(query, k=None):
        """检索：将 query 向量化，在 ChromaDB 中查找最相似的 k 条记忆"""

    def clear():
        """清除：删除 ChromaDB collection"""
```

#### ChromaDB 存储结构

- Collection 名：`companion_memory`
- 每条记忆：`{"user": "...", "ai": "..."}` 拼接为文本
- Metadata：`{"timestamp": ISO 格式, "user_id": "default"}`
- 持久化路径：`settings.chroma_persist_dir`（默认 `./data/chroma`）

---

### 3.6 `src/agent/emotion.py` — 情感感知

**类**：`EmotionDetector`

**5 种情绪**：`happy`、`sad`、`angry`、`anxious`、`neutral`

**核心方法**：`detect(user_input) → str`

```python
def detect(self, user_input: str) -> str:
    # 1. 缓存命中检查（相同输入直接返回）
    if user_input == self._last_input:
        return self._last_emotion
    # 2. 调用 LLM（非流式，temperature=0）→ 只返回标签词
    # 3. 异常降级为 "neutral"
```

**情绪 → 调整指令映射**：

| 情绪 | 指令 |
|------|------|
| `happy` | 分享快乐，语气轻快活泼 |
| `sad` | 温柔安慰，多倾听，不急着给建议 |
| `angry` | 保持冷静同理心，先认同感受 |
| `anxious` | 安抚鼓励，帮助放松 |
| `neutral` | 保持一贯温柔风格 |

**设计要点**：
- 情绪分析用独立 LLM 调用，**不混入对话 LLM 的上下文**，避免 Prompt 污染
- Prompt 设计极简（要求只返回单词），最小化 token 消耗
- 带简单缓存：相同输入直接命中，避免重复分析

---

### 3.7 `src/agent/tools.py` — MCP 工具集成

**类**：
| 类 | 职责 |
|------|------|
| `_MCPTool` | 将单个 MCP 工具封装为 LangChain `BaseTool` |
| `MCPToolManager` | 管理 MCP 服务器连接、工具加载与释放 |

#### MCPToolManager 连接流程

```
connect()
    │
    ├── _build_server_configs()  →  构建服务器列表:
    │       ├── 高德地图 (amap_maps_api_key 非空即启用)
    │       ├── Tavily 搜索 (tavily_api_key 非空即启用)
    │       └── 额外服务器 (mcp_servers_json 文件)
    │
    ├── 逐个连接:
    │       for each config:
    │           ├── stdio_client(params)       → 启动 npx 子进程
    │           ├── ClientSession(read, write) → 建立 MCP 会话
    │           ├── session.initialize()       → 初始化协议
    │           └── session.list_tools()       → 获取工具列表 → 封装为 _MCPTool
    │
    └── 连接失败 → 打印日志 → 跳过该服务器 → 继续下一个
```

#### MCP 工具列表

| 服务 | 工具数 | 典型工具 |
|------|--------|----------|
| 高德地图 (AMap) | 12 | 地理编码、逆地理编码、天气查询、周边搜索、路线规划、IP 定位等 |
| Tavily 搜索 | 1 | 联网搜索、新闻检索 |

#### 关键设计决策

1. **不依赖 langchain-mcp-adapters**：直接用 MCP SDK（`stdio_client` + `ClientSession`）封装，避免依赖版本冲突
2. **连接失败不阻塞**：任何一个 MCP 服务器挂了，不影响其他服务器和核心对话
3. **工具调用跨事件循环处理**：`_MCPTool._run()` 通过 `run_coroutine_threadsafe` 回到原始事件循环，避免 LangChain 同步调用链与 MCP 异步调用链的死锁
4. **Windows 子进程清理**：`disconnect()` 末尾 `await asyncio.sleep(0.1)` 留给 ProactorEventLoop 处理 npx 管道关闭回调

#### 添加新 MCP 工具的方法

1. 在 `.env` 配 API Key（如 `NEW_SERVICE_API_KEY=xxx`）
2. `settings.py` 加对应字段
3. `tools.py` → `_build_server_configs()` 中加配置：

```python
if settings.new_service_api_key:
    configs["new_service"] = {
        "command": "npx",
        "args": ["-y", "package-name"],
        "env": {"API_KEY": settings.new_service_api_key},
    }
```

---

### 3.8 `src/ui/server.py` — FastAPI 后端

**职责**：Web 服务器 + API + 生命周期管理

#### 全局单例

```python
agent = CompanionAgent(use_long_term_memory=True, use_emotion=True)
```

#### 应用生命周期

```
启动:
  1. MCPToolManager().connect()  →  连接所有 MCP 服务器
  2. agent.set_tools(tools)       →  注入工具到 Agent

关闭:
  1. tool_manager.disconnect()    →  断开 MCP、清理 npx 子进程
```

#### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端 HTML |
| GET | `/api/history` | 历史对话 + 当前情绪 |
| POST | `/api/chat` | 发送消息，SSE 流式回复 |
| POST | `/api/reset` | 重置对话（清短期记忆） |
| POST | `/api/clear` | 清除所有数据 |
| GET | `/api/emotion` | 当前情绪 |
| GET | `/api/info` | 伴侣名称和性格 |

#### SSE 流式协议

```
event: emotion
data: {"type":"emotion","emotion":"happy"}

event: chunk
data: {"type":"chunk","content":"今"}

event: chunk
data: {"type":"chunk","content":"天"}

...

event: done
data: {"type":"done"}
```

---

### 3.9 `src/ui/static/index.html` — 前端页面

**技术**：原生 HTML/CSS/JS，零框架依赖

**核心结构**：
- `.chat-container` — 主容器（玻璃拟态）
- `#messages` — 消息列表
- `#chat-form` — 输入框 + 发送按钮
- 情绪标签实时展示
- SSE `ReadableStream` 逐 chunk 渲染 → 打字机效果

**关键交互**：
- Enter 发送，Shift+Enter 换行
- 打字时显示指示器动画（三点跳动）
- 回复完成后自动滚动到底部

---

### 3.10 `src/prompts/companion_prompt.py` — Prompt 模板

提供一个独立函数 `get_companion_prompt()`，但**实际 System Prompt 由 `companion.py:_build_system_prompt()` 内建生成**（含更多动态信息）。此文件保留作为备用/参考。

---

## 四、端到端数据流（一次完整对话）

```
┌─ 浏览器 ────────────────────────────────────────────────────────┐
│  用户输入 "我今天不太开心"                                        │
│  fetch POST /api/chat  →  SSE 连接                                │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ server.py ──────────────────────────────────────────────────────┐
│  POST /api/chat → agent.chat_stream(user_input)                  │
│  → 返回 SSE 流：emotion → chunk... → done                        │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌─ companion.py: chat_stream() ────────────────────────────────────┐
│                                                                  │
│  ┌── 1. 情绪检测 ──────────────────────────────────────────┐     │
│  │   emotion.detect("我今天不太开心") → "sad"                │     │
│  │   → 情绪调整指令："温柔安慰，多倾听，不急着给建议"          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌── 2. 记忆检索 ──────────────────────────────────────────┐     │
│  │   短期记忆(self.history)：最近 20 条消息                   │     │
│  │   长期记忆(memory.retrieve)：ChromaDB 语义检索 k=3 条     │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌── 3. 组装 System Prompt ────────────────────────────────┐     │
│  │   基础人设 + 情绪指令 + 长期记忆 + 工具列表                 │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌── 4. LLM 生成 ──────────────────────────────────────────┐     │
│  │   chain.stream() → yield 逐 chunk                        │     │
│  │   如有 tool_calls → _run_with_tools() → 最多 5 轮        │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌── 5. 记忆更新（流结束后）────────────────────────────────┐     │
│  │   self.history.append(HumanMessage)                       │     │
│  │   self.history.append(AIMessage)                          │     │
│  │   long_term_memory.add_memory(user_input, full_response)  │     │
│  │   _save_history() → chat_history.json                     │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、关键设计决策与取舍

### 5.1 双层记忆架构

**为什么两层？** 单靠 LLM 上下文窗口（短期记忆）无法跨会话，"金鱼记忆"问题严重。引入 ChromaDB 长期记忆后，即使重启也能回忆过去。

| | 短期记忆 | 长期记忆 |
|------|----------|----------|
| 存储方式 | 内存列表 `self.history` | ChromaDB 向量数据库 |
| 生命周期 | 服务关闭即丢失 | 持久化到磁盘 |
| 检索方式 | 全部注入 Prompt（滑动窗口限 20 条） | 语义检索 Top-K（默认 3 条） |
| 用途 | 维持本轮对话连贯性 | 跨会话记忆重要信息 |

### 5.2 流式输出与记忆更新的时序协调

**核心问题**：流式输出期间回复不完整，不能更新记忆。

**解决**：`chat_stream()` 采用"流中缓冲 + 流后统一处理"：
- 流式阶段：`full_response += chunk`，只 yield 不写记忆
- 流结束后：一次性追加 HumanMessage + AIMessage + 写 ChromaDB + JSON 持久化

### 5.3 MCP 工具失败容错

**设计原则**：MCP 工具是增强，不是核心。任何一个挂了不拖累对话。

实现：
- `connect()` 中逐个 try/except，失败服务器跳过
- 工具调用失败 → 将错误信息作为 ToolMessage 返回 LLM → LLM 可以降级为纯文本回复
- 断连时彻底清理所有上下文管理器，避免 npx 子进程泄漏

### 5.4 Embedding 方案演进

| 阶段 | 方案 | 问题 |
|------|------|------|
| 初始 | sentence-transformers 本地模型 | 首次加载慢、CPU 推理耗资源 |
| **当前** | 阿里云 DashScope `text-embedding-v3` | API 调用，速度快、零本地开销 |

`DashScopeEmbeddings` 实现 LangChain `Embeddings` 接口，未来可无缝切换其他 Embedding 服务。

### 5.5 全局单例 Agent

`server.py` 中 `agent` 是全局单例，所有请求共享一份 `self.history` 和 `self.current_emotion`。

**适合当前场景**：单用户 Web UI，一人独用。

**多用户扩展方向**：`LongTermMemory` 已预留 `user_id` 参数，只需将 Agent 实例按 session 隔离即可。

---

## 六、开发与调试指南

### 6.1 首次启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY

# 3. 启动
python main.py
# 访问 http://localhost:8080
```

### 6.2 运行测试

```bash
python test_basic.py              # 基础对话
python test_emotion.py            # 情绪识别
python test_long_term_memory.py   # 长期记忆存取
python test_stream.py             # 流式输出
```

### 6.3 调试技巧

| 场景 | 方法 |
|------|------|
| 看 LLM 实际收到的 Prompt | 在 `_prepare_context` 中 `print(context)` |
| 看 ChromaDB 存储了什么 | 用 `chromadb` 客户端连 `./data/chroma` 查看 collection |
| 看 MCP 工具是否加载 | 启动日志搜索 `[MCP]` 前缀的日志 |
| 看情绪检测结果 | 前端界面情绪标签，或搜索 `agent.current_emotion` |
| 工具被误调用 | 检查 System Prompt 中工具描述是否过于宽泛 |
| 对话卡住 | 检查 DeepSeek API 是否欠费 / tool_calls 是否死循环 |

### 6.4 日志说明

| 日志前缀 | 来源 | 含义 |
|------|------|------|
| `[MCP]` | `tools.py` | MCP 连接、工具加载、失败跳过 |
| 情绪标签 | `emotion.py` | 情绪检测结果 |
| uvicorn 日志 | uvicorn | HTTP 请求记录 |

---

## 七、常见问题排查

### Q1：启动报 `Connection closed` / MCP 工具不工作

**原因**：npx 子进程启动失败（包名错误 / 网络问题 / API Key 缺失）

**排查**：
1. `npx -y tavily-mcp` 手动测是否能启动
2. 检查 `.env` 中对应 API Key 是否填写
3. 不影响核心对话，工具不可用时 LLM 会降级为纯文本回复

### Q2：Ctrl+C 关闭报 `Event loop is closed`

Windows `ProactorEventLoop` 特性。`tools.py` 的 `disconnect()` 已加 `asyncio.sleep(0.1)` 缓解。如仍有此问题，可忽略，不影响功能。

### Q3：长期记忆检索不准确

1. 检查 `DASHSCOPE_API_KEY` 是否有效
2. 调大 `memory_retrieval_k`（`.env` 中 `MEMORY_RETRIEVAL_K=5`）
3. 检查 ChromaDB 是否有数据：`data/chroma/` 是否存在

### Q4：回复过长 / 风格偏离人设

修改 `companion.py:_build_system_prompt()` 中的 Prompt 指令，或参考 BRD.md §11.6 的 Prompt 调优维度。

### Q5：流式输出中断 / 无响应

1. 检查 DeepSeek API 额度
2. 检查 `tool_calls` 是否陷入死循环（日志中看轮次，上限 5 轮）
3. 检查 `_run_with_tools` 中工具执行是否有异常

### Q6：对话历史丢失

检查 `./data/chat_history.json` 文件权限和 JSON 格式是否损坏。损坏时系统会降级为空列表。

---

## 八、扩展指南

### 8.1 添加新的 API 接口

在 `server.py` 中添加路由函数，参考已有接口。

### 8.2 添加新的 MCP 工具

参考 §3.7 "添加新 MCP 工具的方法"。

### 8.3 调整 Prompt / 人设

修改 `companion.py` 中 `_build_system_prompt()` 方法。

### 8.4 替换 LLM

修改 `settings.py` 的 `model_name` / `base_url`，只要是 OpenAI 兼容 API 即可。`llm.py` 无需改动。

### 8.5 替换 Embedding 服务

在 `memory.py` 中实现新的 `Embeddings` 子类，替换 `DashScopeEmbeddings`。`LongTermMemory` 只依赖 `Embeddings` 接口，不耦合具体服务。

---

## 九、技术债务与重构计划

> 以下问题当前影响不大，但规模增长后需处理。记录于此避免遗忘。

| 编号 | 问题 | 当前状态 | 触发条件 | 计划方案 |
|------|------|----------|----------|----------|
| TD-001 | `companion.py` 单一类承载过重 | 🟢 525 行可接受 | 超过 800 行 | 拆分为 `SystemPromptBuilder` + `HistoryManager` + `ToolCallRunner` |
| TD-002 | `server.py` 所有 API 混合在一起 | 🟢 165 行可接受 | API 超过 10 个 | 拆分为 `src/ui/routes/` 独立路由文件 |
| TD-003 | 前端 `index.html` 单文件 HTML/CSS/JS | 🟢 可维护 | 超过 1200 行或引入新交互组件 | 上构建工具（Vite），拆 JS/CSS 为独立文件 |
| TD-004 | `_reverse_geocode()` 放在 Agent 类中 | 🟢 仅一处调用 | 增加更多地理位置逻辑 | 移到 `src/utils/geocoding.py` |
| TD-005 | `data/` 目录散落多个 JSON 文件，无统一数据访问层 | 🟢 文件少可管理 | 数据文件超过 5 个 | 引入 `DataStore` 抽象层统一读写 |
| TD-006 | `clear_all_data()` 不清除用量和日志 | 🟢 设计意图（用量和日志独立生命周期） | 如需完整重置 | 增加 `--full` 选项或独立 API |
| TD-007 | 测试只有 4 个独立脚本，无 pytest 组织 | 🟡 需改进 | 核心逻辑变复杂前 | 迁移到 `tests/` 目录 + pytest fixture |

**下次重构窗口建议**：v0.4 开始前，花半天处理 TD-007（测试组织），其他项到触发条件再动。

---

## 附录：依赖清单

```
# requirements.txt 核心依赖

langchain >= 1.0.0          # Agent 框架
langchain-openai >= 0.3.0   # OpenAI/DeepSeek 接口
fastapi >= 0.115.0          # Web 框架
uvicorn >= 0.34.0           # ASGI 服务器
pydantic-settings >= 2.0    # 配置管理
chromadb >= 0.5.0           # 向量数据库
mcp >= 1.0.0                # MCP 协议 SDK
httpx >= 0.25               # HTTP 客户端（Embedding API 调用）
```

---

*最后更新：2026-07-09*
