# 简历项目经历：AI 虚拟伴侣（AI-Companion）

## 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | AI 虚拟伴侣（AI-Companion）|
| 时间 | 2026.06 - 2026.07 |
| 角色 | 独立全栈开发 |
| 代码量 | ~2500 行 Python + ~1200 行 JavaScript + 27 项自动化测试 |
| 版本 | 6 个迭代版本，26 项功能需求 |

---

## 简历版本（一页纸）

### AI 虚拟伴侣 — 有记忆、有情感、有工具调用能力的个人 AI 助手系统

**技术栈**：Python / FastAPI / LangChain / ChromaDB / DeepSeek API / MCP 协议 / 原生 JavaScript

- 设计并实现了**双层记忆架构**（滑动窗口短期记忆 + ChromaDB 向量长期记忆），引入**时间衰减算法**（指数衰减函数 `exp(-age/half_life)`）对语义检索结果加权重排，解决"记忆只增不减导致检索质量下降"的问题
- 基于 **MCP 协议**（Model Context Protocol）集成 3 个外部工具服务器共 19 个工具（高德地图 12 个、Tavily 搜索 5 个、time server 2 个），通过 `stdio_client` 子进程模式 + `asyncio.run_coroutine_threadsafe` 解决跨事件循环死锁问题，LLM 自主决策调用，支持最多 5 轮工具链（Chain-of-Tool）
- 实现了**多源 System Prompt 动态组装**：基础人设 + 情绪感知调整 + 长期记忆 + 用户画像 + 位置城市 + 工具列表 → 运行时逐段拼接，LLM 调用前一次完成上下文准备
- 自研 **LangChain Tool**（`schedule_event`），LLM 借助 mcp-server-time 计算日期后直接调用写入日历，并自动生成 **iCalendar (.ics) 标准格式**文件导入系统日历实现跨进程提醒
- 设计了**每轮对话 4 次 LLM 调用流水线**：情绪检测（非流式）→ 主对话生成（SSE 流式，可选工具调用）→ 用户画像提取（结构化 JSON 输出）→ Embedding 向量化写入，通过 UsageTracker 全程统计 Token 消耗与成本估算
- 编写 **27 项 pytest 单元测试**，采用 `unittest.mock` 完全 Mock DeepSeek API，覆盖对话生成、情绪识别、用量统计三条核心链路，10 秒零费用回归验证
- 完成前后端全栈开发：FastAPI SSE 流式推送 + 原生 JavaScript `ReadableStream` 解析 + `marked.js` Markdown 实时渲染 + `highlight.js` 代码语法高亮 + `SpeechRecognition` 语音输入 + 多文件文档上传（PDF/Word/TXT 解析注入 LLM 上下文）
- 建立了**用量监控与成本控制系统**（字符级 Token 估算 + DeepSeek 费率先行定价 + 每日预算上限自动拦截）和**结构化日志体系**（`TimedRotatingFileHandler` 按天切割，30 天保留）

---

## 面试扩展问答

### Q1：双层记忆具体怎么做到的？

**短期记忆**：LangChain `MessagesPlaceholder`，内存中保留最近 20 条消息（HumanMessage + AIMessage），超出滑动窗口自动截断。

**长期记忆**：每轮对话结束后，将"用户消息 + AI 回复"拼接为文本，通过阿里云 DashScope `text-embedding-v3` 向量化（768 维）存入 ChromaDB。下次对话时语义检索 top-k 条记忆。

**时间衰减**：检索时取 k×3 条候选，用 `similarity_search_with_score` 获取语义相似度分，再与时间衰减因子 `exp(-age_seconds / half_life_seconds)` 相乘（半衰期默认 30 天），重排后取 top-k。越久远的记忆权重越低，但不完全遗忘。

### Q2：MCP 工具是怎么集成进来的？

不依赖 `langchain-mcp-adapters`（避免版本冲突），直接用 MCP SDK 的 `stdio_client` + `ClientSession`：

1. `_build_server_configs()` → 根据 `.env` 中 API Key 是否配置决定启用哪些服务器
2. `connect()` → `npx/python -m` 启动子进程 → `session.initialize()` → `session.list_tools()`
3. 每个工具封装为 `_MCPTool(BaseTool)`，用 `pydantic.create_model` 动态生成 args_schema
4. `agent.llm.bind_tools(tools)` → LLM 在 stream 中返回 `tool_calls`
5. `_run_with_tools()` 循环检测 `response.tool_calls` → 执行工具 → 回填 `ToolMessage` → 重新 generate，最多 5 轮

**关键坑点**：Windows `ProactorEventLoop` 下，LangChain 的同步 `invoke` 不能直接调 MCP 的异步 `session.call_tool`，会死锁。解决方案：`_MCPTool._run()` 通过 `asyncio.run_coroutine_threadsafe` 把协程调度回原始事件循环执行。

### Q3：System Prompt 里总共注入了几种上下文？

5 种，运行时动态拼接：

| 顺序 | 上下文 | 触发条件 |
|------|--------|----------|
| 1 | 基础人设 | 始终 |
| 2 | 情绪调整指令 | 启用情感感知 |
| 3 | 长期记忆文本 | 检索到记忆 |
| 4 | 用户画像（名字/年龄/位置/职业等） | 有画像数据 |
| 5 | 日程提醒列表 | 有未来日程 |
| 6 | 位置城市 | 前端传了 GPS 坐标 |
| 7 | 工具列表名称 | 有 MCP 工具 |

### Q4：用户画像是怎么自动构建的？

**提取**：每轮对话后，将 `(user_input, ai_response)` 发给 LLM（非流式独立调用），要求返回结构化 JSON（8 个维度：名字、年龄、位置、职业、兴趣、技能、状态、其他）。无新信息返回 `null`。

**存储**：`data/user_profile.json`，新信息逐字段比对去重，"住在北京" → "搬到上海" 自动覆盖。

**注入**：System Prompt 追加 `【用户画像】- 名字：张三 - 所在地：北京 …`，LLM 在对话中自然引用。

### Q5：Token 用量怎么统计的？

不依赖 API 返回的 `usage` 字段（SSE 流式调用无此字段）。采用**字符级估算**：中文 1.5 字/token、英文/符号 3.5 字/token，误差 ±10%。

`UsageTracker` 记录每次 LLM 调用的输入/输出 token 数，累加到 `data/usage.json`，按天归零。费用按 DeepSeek v4-flash 定价计算（输入 ¥0.55/1M、输出 ¥2.19/1M）。

**预算控制**：每次对话前 `check_budget()` 检查，超限直接返回提示文案，不调用 LLM。

### Q6：前端怎么做的，为什么不用框架？

当时选原生 HTML/CSS/JS 的原因：项目初期快速验证，零构建工具链。一个 `index.html` 文件包含全部 CSS + JS。

技术点：
- **SSE 流式**：`fetch` → `resp.body.getReader()` → 逐 chunk 解析 `data:` 前缀
- **Markdown 渲染**：`marked.js` 配置 GFM + `highlight.js` 代码高亮，流式期间 `textContent` 避免闪烁，完成后切 `innerHTML`
- **语音输入**：`SpeechRecognition` API（Edge/Chrome 内置），`lang='zh-CN'`，实时填入输入框
- **位置感知**：`navigator.geolocation` 每消息附带 `{lat, lng}`，后端高德 REST API 逆地理编码
- **文档上传**：`FormData` → `POST /api/upload` → `pdfplumber`/`python-docx` 解析 → 注入 LLM 上下文

---

## 项目结构一览

```
AI-Companion/
├── main.py                          # 入口：uvicorn + 后台线程
├── src/
│   ├── config/settings.py           # Pydantic Settings，20+ 配置项
│   ├── agent/
│   │   ├── companion.py             # Agent 主类：编排对话全流程
│   │   ├── llm.py                   # ChatOpenAI 封装 DeepSeek
│   │   ├── memory.py                # ChromaDB + 时间衰减检索
│   │   ├── emotion.py               # 5 情绪 LLM 分类 + 自适应指令
│   │   ├── tools.py                 # MCP 工具管理器 + _MCPTool 封装
│   │   ├── usage.py                 # Token 估算 + 费用统计 + 预算控制
│   │   ├── profile.py               # 画像提取（LLM JSON 结构化输出）
│   │   └── calendar.py              # 日程 CRUD + .ics 生成 + LangChain Tool
│   ├── utils/logger.py              # TimedRotatingFileHandler 日志
│   └── ui/
│       ├── server.py                # FastAPI：10 个 API + SSE + 生命周期
│       └── static/index.html        # 前端：玻璃拟态 UI，零框架
├── tests/                           # 27 项 pytest + Mock LLM
├── data/                            # 运行时数据（JSON/ChromaDB/日志/.ics）
└── BRD.md / DESIGN.md / TECH.md     # 完整产品 + 技术 + 架构文档
```
