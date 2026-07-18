# 简历项目经历：AI 虚拟助手（AI-Companion）

## 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | AI 虚拟助手（AI-Companion）|
| 时间 | 2026.06 - 2026.07 |
| 角色 | 独立全栈开发 |
| 代码量 | ~2500 行 Python + ~1200 行 JavaScript + 27 项自动化测试 |
| 版本 | 6 个迭代版本，26 项功能需求 |

---

## 项目背景

个人 AI 助手市场被闭源产品垄断，用户数据在第三方服务器，无法定制人设和能力。自己从零造一个完全可控的 AI 助手——数据本地化、性格自定义、能力可扩展。6 个迭代版本渐进交付，从"能聊天"逐步叠加记忆、情感、工具、画像等能力。

---

## 简历要点

**技术栈**：Python / FastAPI / LangChain / ChromaDB / DeepSeek / MCP 协议 / 原生 JavaScript

- 设计**双层记忆架构**（滑动窗口短期 + ChromaDB 向量长期），引入**时间衰减算法** `exp(-age/half_life)` 对语义检索结果加权，解决记忆膨胀导致检索质量下降的问题
- 集成 **MCP 协议**接入 3 个外部服务器共 19 个工具（高德地图 12、Tavily 搜索 5、time server 2），`stdio_client` 子进程模式 + `run_coroutine_threadsafe` 解决 Windows 跨事件循环死锁，LLM 自主决策调用，最多 5 轮工具链
- 实现**多源 System Prompt 动态组装**：基础人设 → 情绪调整 → 长期记忆 → 用户画像 → 位置城市 → 工具列表，运行时逐段拼接
- 自研 **LangChain Tool**（`schedule_event`），LLM 借助 time server 计算日期后写入 JSON 日历，同时生成 `.ics` 标准文件供导入系统日历
- 编写 **27 项 pytest**，Mock LLM 全链路，覆盖对话/情绪/用量三条核心链路，10 秒零费用回归
- 全栈开发：FastAPI SSE 流式 + 原生 JS `ReadableStream` + `marked.js` Markdown 渲染 + `SpeechRecognition` 语音输入 + PDF/Word/TXT 多文件上传
- 建立**用量监控**（字符级 Token 估算 + 每日预算上限）和**结构化日志**（按天切割，30 天保留）

---

## 面试问答

### Q1：双层记忆怎么实现？

**短期**：`MessagesPlaceholder`，内存保留 20 条，滑动窗口截断。**长期**：DashScope `text-embedding-v3` 向量化 → ChromaDB 语义检索 k×3 候选 → 时间衰减 `exp(-age/half_life)` 加权 → 重排取 top-k。半衰期 30 天，越久权重越低。

### Q2：MCP 工具如何集成？

不依赖 `langchain-mcp-adapters`。`stdio_client` + `ClientSession` → 子进程启动 → `list_tools()` → 每个工具封装为 `_MCPTool(BaseTool)` → `bind_tools(tools)` → LLM 返回 `tool_calls` → 执行 → `ToolMessage` 回填 → 最多 5 轮。

**关键坑**：Windows `ProactorEventLoop` 下同步 `invoke` 不能直接调异步 `session.call_tool`，会死锁。用 `run_coroutine_threadsafe` 调度回原始事件循环。

### Q3：System Prompt 注入哪些上下文？

7 种：基础人设 → 情绪调整指令 → 长期记忆 → 可用工具列表 → 用户画像 → 已有日程 → 位置城市。运行时按条件逐段拼接。

### Q4：用户画像怎么提取？

每轮对话后 LLM 非流式调用，返回 8 维 JSON（名字/年龄/位置/职业/兴趣/技能/状态/其他）。逐字段比对去重，自动覆盖旧信息。注入 System Prompt `【用户画像】`。

### Q5：Token 用量怎么统计？

SSE 流式不返回 `usage` 字段。用字符估算：中文 1.5 字/token、英文 3.5 字/token，误差 ±10%。`UsageTracker` 记录 + `data/usage.json` 持久化 + 按天归零 + 预算超限拦截。
