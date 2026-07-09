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

import gc
import logging
import signal
import uvicorn
from src.ui.server import app
from src.utils.logger import setup_logging

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("main")

    port = int(os.environ.get("PORT", 8080))
    logger.info("=" * 50)
    logger.info("  AI 虚拟伴侣 — 启动中...")
    logger.info("  监听端口: %d", port)
    logger.info("=" * 50)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info",
                            timeout_graceful_shutdown=10)
    server = uvicorn.Server(config)

    def _shutdown(_sig, _frame):
        logger.info("正在关闭服务...")
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.run()
    finally:
        logger.info("服务已停止")
        gc.collect()


