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
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response, FileResponse
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
    location: Optional[dict] = None
    documents: Optional[list] = None  # [{filename, text}]

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
        for piece in agent.chat_stream(user_input, location=req.location, documents=req.documents):
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

@app.get("/api/emotion/history")
async def emotion_history(days: int = 7):
    """获取情绪历史（按天聚合）"""
    return JSONResponse(agent.get_emotion_history(days=days))

@app.get("/api/calendar/{event_id}.ics")
async def download_ics(event_id: str):
    """下载日程的 .ics 文件（双击导入系统日历）"""
    path = agent.calendar.get_ics_path(event_id)
    if not path:
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return FileResponse(path, media_type="text/calendar", filename=f"{event_id}.ics")

@app.get("/api/calendar")
async def get_calendar():
    """获取未来日程列表"""
    return JSONResponse(agent.calendar.list_events(upcoming_only=True))

@app.post("/api/calendar")
async def add_calendar(req: dict):
    """添加日程"""
    evt = agent.calendar.add_event(
        req.get("title", ""),
        req.get("datetime", ""),
        req.get("notes", ""),
    )
    return JSONResponse(evt)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传并解析文档（PDF/Word/TXT/Markdown），返回纯文本"""
    import tempfile
    filename = (file.filename or "").lower()
    content = await file.read()

    try:
        if filename.endswith(".pdf"):
            import pdfplumber
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content); tmp.flush()
                with pdfplumber.open(tmp.name) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            os.unlink(tmp.name)
            return JSONResponse({"text": text[:50000], "filename": file.filename})

        elif filename.endswith(".docx"):
            import docx
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content); tmp.flush()
                doc = docx.Document(tmp.name)
                text = "\n".join(p.text for p in doc.paragraphs)
            os.unlink(tmp.name)
            return JSONResponse({"text": text[:50000], "filename": file.filename})

        elif filename.endswith((".txt", ".md", ".markdown")):
            text = content.decode("utf-8", errors="replace")
            return JSONResponse({"text": text[:50000], "filename": file.filename})

        else:
            return JSONResponse({"error": "不支持的文件格式，仅支持 PDF/Word/TXT/Markdown"}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": f"文件解析失败: {str(e)}"}, status_code=500)

@app.get("/api/personas")
async def get_personas():
    """获取所有人设预设"""
    import json as _json
    path = os.path.join(os.path.dirname(settings.chat_history_file), "personas.json")
    if not os.path.exists(path):
        return JSONResponse([])
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(_json.load(f))

@app.post("/api/persona/switch")
async def switch_persona(req: dict):
    """切换人设"""
    persona_id = req.get("id", "")
    path = os.path.join(os.path.dirname(settings.chat_history_file), "personas.json")
    if os.path.exists(path):
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            personas = _json.load(f)
        for p in personas:
            if p["id"] == persona_id:
                agent.apply_persona(p["name"], p["personality"], p["backstory"])
                return JSONResponse({"status": "ok", "name": p["name"]})
    return JSONResponse({"status": "error", "message": "未找到该人设"}, status_code=400)

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
