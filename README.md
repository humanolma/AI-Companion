# AI 虚拟伴侣 / AI Virtual Companion

一个有记忆、有情感、有能力的个人 AI 助手。基于 LangChain + DeepSeek，支持 MCP 工具集成、位置感知、用量监控。

**当前版本：v0.8**

## ✨ 功能特点

### 核心能力
- 🧠 **双层记忆**：短期上下文 + ChromaDB 长期记忆（语义检索 + 时间衰减）
- 👤 **用户画像**：LLM 自动从对话中提取个人信息，越用越懂你
- 🎭 **5套人设**：小梦 / 知心姐姐 / 幽默死党 / 知性导师 / 元气少女，一键切换
- 💖 **情感感知**：5 种情绪识别，动态调整语气，情绪趋势柱状图
- ⌨️ **流式输出**：SSE 打字机 + Markdown 渲染 + 代码语法高亮
- 🎨 **响应式 UI**：玻璃拟态设计，暗色/亮色主题跟随系统

### 工具与感知
- 🗺️ **MCP 工具**：高德地图(12) + Tavily 搜索 + time server
- 📍 **位置感知**：GPS → 逆地理编码 → 自动知道你在哪个城市
- 📅 **日程创建**：自然语言"提醒我后天开会" → LLM 自动写入日历
- 🔥 **用量监控**：Token 统计 + 费用估算 + 每日预算上限

### 交互体验
- 🎤 **语音输入**：浏览器 SpeechRecognition，说完自动填入
- 📎 **文档上传**：PDF/Word/TXT 多文件，上传后直接提问
- `/` **快捷指令**：`/天气` `/搜索` `/导出` `/清空` `/重置`
- 🔍 **历史搜索**：关键词搜索 + 点击跳转到对应消息
- 📥 **对话导出**：Markdown / JSON 一键下载

### 工程化
- 📝 **日志**：按天切割，30 天保留
- 🧪 **测试**：27 项 pytest，Mock LLM，0 费用
- ⌨️ **快捷键**：Enter 发送 / Shift+Enter 换行 / Ctrl+L 清空

## 📦 安装

### 1. 环境准备

```bash
conda create -n ai-companion python=3.10
conda activate ai-companion
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，至少配置：

```ini
# 必填
DEEPSEEK_API_KEY=sk-xxx          # LLM（DeepSeek）
DASHSCOPE_API_KEY=sk-xxx         # Embedding（阿里云 DashScope）

# 可选（不填则对应功能自动禁用）
AMAP_MAPS_API_KEY=xxx            # 高德地图工具
TAVILY_API_KEY=tvly-xxx          # 联网搜索
DAILY_BUDGET_LIMIT=0             # 每日预算上限（元），0=不限
```

## 🚀 运行

```bash
# 1. 启动后端 API
python main.py

# 2. 打开前端（选一种）
# 方式A：直接双击 frontend/index.html
# 方式B：npx serve frontend -p 3000
```

前端是独立静态文件，后端是纯 API。前端 `API_BASE` 变量指向 `http://localhost:8080`。

## ⚙️ 主要配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `COMPANION_NAME` | 小梦 | 伴侣名字 |
| `COMPANION_PERSONALITY` | 温柔、善解人意、有点俏皮 | 性格描述 |
| `COMPANION_BACKSTORY` | （空） | 背景故事 |
| `MODEL_NAME` | deepseek-v4-flash | LLM 模型 |
| `TEMPERATURE` | 0.7 | 生成温度 |
| `MEMORY_RETRIEVAL_K` | 3 | 长期记忆每次检索条数 |
| `MAX_SHORT_TERM_HISTORY` | 20 | 短期记忆最大条数 |
| `DAILY_BUDGET_LIMIT` | 0 | 每日费用上限（元） |
| `DEEPSEEK_INPUT_PRICE` | 0.55 | 输入价格（元/百万 tokens） |
| `DEEPSEEK_OUTPUT_PRICE` | 2.19 | 输出价格（元/百万 tokens） |

## 🏗️ 项目结构

```
AI-Companion/
├── main.py                         # [入口] uvicorn 启动
├── requirements.txt
├── .env.example
│
├── BRD.md                          # 业务需求说明书
├── DESIGN.md                       # 技术设计文档
├── README.md
│
├── src/
│   ├── config/
│   │   └── settings.py             # Pydantic Settings 配置管理
│   │
│   ├── agent/                      # [核心] Agent 层
│   │   ├── companion.py            # CompanionAgent 主类：对话编排
│   │   ├── llm.py                  # LLM 初始化（DeepSeek / OpenAI 兼容）
│   │   ├── memory.py               # 长期记忆（ChromaDB + DashScope Embedding）
│   │   ├── emotion.py              # 情感感知（LLM 情绪分析）
│   │   ├── tools.py                # MCP 工具管理（高德 + Tavily）
│   │   └── usage.py                # 用量追踪（Token 估算 + 费用统计）
│   │
│   ├── utils/
│   │   └── logger.py               # 统一日志配置（按天切割）
│   │
│   └── ui/                         # [前端] Web 交互层
│       ├── server.py               # FastAPI（REST + SSE）
│       └── static/
│           ├── index.html          # 前端页面（玻璃拟态 UI）
│           └── avatar.jpg          # 头像
│
├── assets/
│   └── companion_avatar.svg
│
├── data/                           # 运行时数据（.gitignore）
│   ├── chat_history.json           # 对话持久化
│   ├── usage.json                  # 用量统计
│   ├── mcp_servers.json            # 额外 MCP 配置
│   ├── chroma/                     # ChromaDB 向量数据库
│   └── logs/                       # 日志文件（按天切割）
│
├── test_basic.py
├── test_emotion.py
├── test_long_term_memory.py
└── test_stream.py
```

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek API（deepseek-v4-flash）| OpenAI 兼容，性价比高 |
| Agent | LangChain 1.x + MCP SDK | Prompt 模板、流式输出、工具调用循环 |
| 长期记忆 | ChromaDB | 向量数据库，语义检索 |
| Embedding | 阿里云 DashScope text-embedding-v3 | 云端推理，零本地开销 |
| MCP 工具 | 高德地图 + Tavily 搜索 | npx 子进程，连接失败自动降级 |
| 后端 | FastAPI | REST API + SSE 流式 |
| 前端 | 原生 HTML/CSS/JS + marked.js + highlight.js | 玻璃拟态、Markdown 渲染、GPS 定位 |
| 日志 | logging + TimedRotatingFileHandler | 按天切割，7 天保留 |
| 配置 | pydantic-settings | `.env` 环境变量管理 |

## 🧪 测试

```bash
python test_basic.py               # 基础对话
python test_long_term_memory.py    # 长期记忆存取
python test_emotion.py             # 情绪识别
python test_stream.py              # 流式输出
```

## 📐 架构设计

### 数据流

```
浏览器（GPS位置 + 消息）
  → FastAPI SSE
  → EmotionDetector（情绪分析）
  → LongTermMemory（语义检索）
  → System Prompt 组装（人设 + 情绪 + 记忆 + 位置 + 日期 + 工具列表）
  → LLM 生成（可选 MCP 工具调用循环）
  → UsageTracker（记录用量）
  → SSE 流式推送 + Markdown 渲染
  → 更新记忆 + 持久化
```

### 关键设计决策

- **双层记忆**：短期（内存列表 20 条）+ 长期（ChromaDB 语义检索 3 条）
- **MCP 失败容错**：工具是增强不是必须，任何一个挂掉不拖累对话
- **逆地理编码走 REST**：不用 MCP 工具链，避免异步复杂度
- **Token 字符估算**：不依赖 API 返回（流式无此字段），中文/1.5 + 英文/3.5
- **System Prompt 多源注入**：情绪调整 + 长期记忆 + 位置城市 + 工具列表 → 运行时分段拼接

## 🗺️ 版本规划

| 版本 | 状态 | 范围 |
|------|------|------|
| **v0.1** | ✅ | Phase 1：核心对话引擎（LLM/Memory/Emotion/SSE/UI） |
| **v0.2** | ✅ | Markdown 渲染、位置感知、日志系统、快捷键、用量监控 |
| **v0.3** | ✅ | 日期感知（MCP time）、对话导出、历史搜索、用户画像 |
| **v0.4** | ✅ | 代码清理、单元测试(27项)、错误降级、记忆衰减、前端布局 |
| **v0.5** | ✅ | 快捷指令、日程创建、情绪趋势、响应式主题 |
| **v0.6** | ✅ | 语音输入、5套人设、多文件上传 |
| **v0.7** | ✅ | 前端 UI 重设计（暖色毛玻璃 + 双主题） |
| **v0.8** | ✅ | 前后端分离（CORS + 独立前端目录） |
| **v1.0** | 🚧 | 认证 + Docker + 公网上线 |
| **v1.0** | 🚧 | 认证 + Docker + Health + 公网上线 |

详见 [BRD.md](BRD.md) 和 [DESIGN.md](DESIGN.md)。

## 📄 许可证

MIT License
