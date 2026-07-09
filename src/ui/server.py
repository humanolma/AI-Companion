"""
FastAPI 后端 — 提供 REST API + SSE 流式接口

接口：
  GET  /                  → 前端页面
  GET  /api/history       → 获取历史对话
  POST /api/chat          → 发送消息（SSE 流式回复）
  POST /api/reset         → 重置当前对话
  POST /api/clear         → 清除所有数据
  GET  /api/emotion       → 获取当前情绪
"""
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.agent.companion import CompanionAgent
from src.agent.tools import MCPToolManager
from src.config.settings import settings

# 创建 Agent 实例
os.makedirs(settings.chroma_persist_dir, exist_ok=True)
# ========== 全局单例 Agent ==========
agent = CompanionAgent(use_long_term_memory=True, use_emotion=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时连接 MCP 工具并注入 Agent，关闭时释放"""
    tool_manager = MCPToolManager()
    await tool_manager.connect()
    agent.set_tools(tool_manager.get_tools())
    yield
    await tool_manager.disconnect()


app = FastAPI(title="AI Virtual Companion", lifespan=lifespan)

# 静态文件目录
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/history")
async def get_history():
    """获取历史对话"""
    return JSONResponse({
        "history": agent.get_history_display(),
        "emotion": agent.current_emotion,
    })

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """流式对话接口（SSE）"""
    user_input = req.message.strip()
    if not user_input:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    def event_stream():
        # 先发送情绪信息
        emotion_data = json.dumps({
            "type": "emotion",
            "emotion": agent.current_emotion,
        }, ensure_ascii=False)
        yield f"data: {emotion_data}\n\n"

        # 流式发送回复
        for piece in agent.chat_stream(user_input):
            chunk_data = json.dumps({
                "type": "chunk",
                "content": piece,
            }, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"

        # 发送结束信号
        done_data = json.dumps({"type": "done"}, ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/api/reset")
async def reset():
    """重置当前对话（不删文件）"""
    agent.reset()
    return JSONResponse({"status": "ok", "message": "对话已重置"})

@app.post("/api/clear")
async def clear_all():
    """清除所有数据"""
    agent.clear_all_data()
    return JSONResponse({"status": "ok", "message": "所有数据已清除"})

@app.get("/api/info")
async def info():
    """获取伴侣信息"""
    return JSONResponse({
        "name": settings.companion_name,
        "personality": settings.companion_personality,
    })
