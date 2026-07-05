"""
AI 虚拟伴侣 — 入口文件

启动方式：
    python main.py

然后浏览器打开 http://127.0.0.1:8000
"""
import uvicorn
from src.ui.server import app

if __name__ == "__main__":
    print("=" * 50)
    print("  AI 虚拟伴侣 — 启动中...")
    print("  浏览器打开: http://127.0.0.1:8000")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000)


