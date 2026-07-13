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
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from src.agent.companion import CompanionAgent
from src.agent.tools import MCPToolManager
from src.agent.usage import UsageTracker
from src.config.settings import settings

# ========== 全局单例 ==========
os.makedirs(settings.chroma_persist_dir, exist_ok=True)
usage_tracker = UsageTracker(
    data_file=settings.usage_data_file,
    budget=settings.daily_budget_limit,
    input_price=settings.deepseek_input_price,
    output_price=settings.deepseek_output_price,
)
agent = CompanionAgent(
    use_long_term_memory=True,
    use_emotion=True,
    usage_tracker=usage_tracker,
)


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
    location: Optional[dict] = None  # {lat: number, lng: number}

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
        for piece in agent.chat_stream(user_input, location=req.location):
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

@app.get("/api/export")
async def export_chat(format: str = "markdown"):
    """导出对话历史（Markdown 或 JSON）"""
    history = agent.get_history_display()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if format == "json":
        return JSONResponse({
            "exported_at": now,
            "companion": settings.companion_name,
            "messages": history,
        })

    # Markdown 格式
    lines = [
        f"# {settings.companion_name} · 对话记录",
        f"> 导出时间：{now}",
        "",
        "---",
        "",
    ]
    for msg in history:
        role_label = "## 我" if msg["role"] == "user" else f"## {settings.companion_name}"
        lines.append(role_label)
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    md_content = "\n".join(lines)

    return Response(
        content=md_content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="companion_chat_{now[:10]}.md"',
        },
    )

@app.get("/api/search")
async def search_chat(q: str = ""):
    """搜索对话历史，返回匹配的消息列表"""
    keyword = q.strip()
    if not keyword:
        return JSONResponse({"results": [], "total": 0})

    history = agent.get_history_display()
    results = []
    for i, msg in enumerate(history):
        if keyword.lower() in msg["content"].lower():
            # 截取关键词前后 40 字作为预览
            idx = msg["content"].lower().index(keyword.lower())
            start = max(0, idx - 40)
            end = min(len(msg["content"]), idx + len(keyword) + 40)
            preview = msg["content"][start:end]
            if start > 0:
                preview = "…" + preview
            if end < len(msg["content"]):
                preview = preview + "…"

            results.append({
                "index": i,
                "role": msg["role"],
                "preview": preview,
                "full_content": msg["content"],
                "keyword": keyword,
            })

    return JSONResponse({"results": results, "total": len(results)})

@app.get("/api/health")
async def health():
    """健康检查：LLM 连通性 + MCP 工具状态"""
    import time as _time
    start = _time.time()
    alive = True
    try:
        agent.llm.invoke("ping")
    except Exception:
        alive = False
    latency = round((_time.time() - start) * 1000)

    return JSONResponse({
        "status": "ok" if alive else "degraded",
        "llm": alive,
        "llm_latency_ms": latency,
        "mcp_tools": len(agent.tools),
        "memory_count": len(agent.history),
    })

@app.get("/api/usage")
async def get_usage():
    """获取今日用量统计"""
    return JSONResponse(usage_tracker.get_stats())

@app.get("/api/profile")
async def get_profile():
    """获取用户画像"""
    return JSONResponse(agent.user_profile.get_profile())

@app.get("/api/info")
async def info():
    """获取伴侣信息"""
    return JSONResponse({
        "name": settings.companion_name,
        "personality": settings.companion_personality,
    })
