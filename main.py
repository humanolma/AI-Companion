"""
AI 虚拟伴侣 — 入口文件

启动方式：
    python main.py

然后浏览器打开 http://127.0.0.1:8080
"""
# 抑制第三方库的进度条和调试输出
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

import uvicorn
from src.ui.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("=" * 50)
    print("  AI 虚拟伴侣 — 启动中...")
    print(f"  监听端口: {port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)


