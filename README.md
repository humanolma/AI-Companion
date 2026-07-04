# AI 虚拟伴侣 / AI Virtual Companion

一个基于 LangChain + DeepSeek 的 AI 虚拟伴侣，有记忆、有情感、会聊天。

## ✨ 功能特点

- 🧠 **记忆系统**：短期对话上下文 + ChromaDB 长期记忆，跨会话也能记住你说过的话
- 🎭 **角色人设**：可自定义伴侣的性格、背景故事、说话风格
- 💖 **情感感知**：识别用户情绪（开心/低落/生气/焦虑/平静），动态调整回复语气
- ⌨️ **流式输出**：SSE 流式推送，打字机效果实时反馈
- 🎨 **自研 Web UI**：FastAPI 后端 + 原生 HTML/CSS/JS 前端，玻璃拟态设计
- 💾 **对话持久化**：JSON 本地存储，重启不丢失对话记录

## 📦 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/ai-companion.git
cd ai-companion
```

2. 创建 conda 环境
```bash
conda create -n ai-companion python=3.10
conda activate ai-companion
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

## 🚀 运行

```bash
python main.py
```

然后在浏览器打开 `http://127.0.0.1:8000`

## ⚙️ 配置

编辑 `.env` 文件自定义：
- `DEEPSEEK_API_KEY`：DeepSeek API 密钥
- `COMPANION_NAME`：伴侣名字（默认：小梦）
- `COMPANION_PERSONALITY`：性格特点（默认：温柔、善解人意、有点俏皮）
- `COMPANION_BACKSTORY`：背景故事
- `MODEL_NAME`：使用的模型（默认：deepseek-chat）
- `TEMPERATURE`：温度参数（0-1，越大越有创意）

## 🏗️ 项目结构

```
ai-companion/
├── main.py                    # 入口文件（uvicorn 启动 FastAPI）
├── requirements.txt
├── .env.example
├── src/
│   ├── config/
│   │   └── settings.py        # Pydantic Settings 配置管理
│   ├── agent/
│   │   ├── llm.py             # LLM 初始化（DeepSeek / OpenAI 兼容）
│   │   ├── companion.py       # 虚拟伴侣 Agent 主类
│   │   ├── memory.py          # 长期记忆（ChromaDB + 本地 Embedding）
│   │   └── emotion.py         # 情感感知模块
│   ├── prompts/
│   │   └── companion_prompt.py # Prompt 模板
│   └── ui/
│       ├── server.py          # FastAPI 后端（REST + SSE）
│       └── static/
│           ├── index.html     # 前端页面（玻璃拟态设计）
│           └── avatar.jpg     # 伴侣头像
├── test_basic.py              # 基础对话测试
├── test_long_term_memory.py   # 长期记忆测试
├── test_emotion.py            # 情感感知测试
└── test_stream.py             # 流式输出测试
```

## 🛠️ 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek API | OpenAI 兼容接口，成本低 |
| Agent 框架 | LangChain 1.x | Prompt 模板、链式调用、流式输出 |
| 长期记忆 | ChromaDB | 向量数据库，语义检索历史对话 |
| Embedding | sentence-transformers | 本地模型 all-MiniLM-L6-v2，无需额外 API |
| 后端 | FastAPI | REST API + SSE 流式接口 |
| 前端 | 原生 HTML/CSS/JS | 玻璃拟态 UI，无框架依赖 |
| 配置管理 | pydantic-settings | 类型安全的环境变量管理 |

## 🧪 测试

```bash
# 基础对话测试
python test_basic.py

# 长期记忆测试（跨会话记忆检索）
python test_long_term_memory.py

# 情感感知测试（5 种情绪检测）
python test_emotion.py

# 流式输出测试（打字机效果）
python test_stream.py
```

## 📐 架构亮点

**记忆系统双层架构**：
- **短期记忆**：LangChain MessagesPlaceholder，保留最近 N 轮对话上下文
- **长期记忆**：ChromaDB 向量数据库，每次对话自动存储 + 语义检索相关历史

**情感感知闭环**：
用户输入 → LLM 情绪分析 → 识别 5 种情绪 → 动态注入情绪调整指令到 System Prompt → 生成匹配语气的回复

**自实现 Embedding 适配层**：
DeepSeek 不提供 Embeddings API，自实现 `LocalEmbeddings` 类继承 LangChain 接口，包装 sentence-transformers 本地模型，无缝接入 ChromaDB。

## 📄 许可证

MIT License
