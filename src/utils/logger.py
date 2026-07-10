"""
统一日志配置 — 控制台 + 按天切割文件

用法：
    from src.utils.logger import setup_logging
    import logging

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("something happened")
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logging(level: int = logging.INFO, log_dir: str = "./data/logs"):
    """初始化日志系统：控制台 INFO+，文件 DEBUG+，按天切割保留 7 天"""
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台 — INFO 及以上
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 文件 — DEBUG 及以上，每天午夜切割，保留 30 天
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "companion"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=False,
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 降低第三方库日志噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
