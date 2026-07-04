from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置类（从环境变量 + .env 文件读取）"""
    
    # === API 配置 ===
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    
    # === 模型配置 ===
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    
    # === Embedding 配置（用于长期记忆）===
    # DeepSeek 不支持 embeddings，使用本地 sentence-transformers
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers 模型
    embedding_device: str = "cpu"  # "cpu" or "cuda"
    
    # === 记忆配置 ===
    chroma_persist_dir: str = "./data/chroma"
    memory_retrieval_k: int = 3  # 每次检索返回的最相关记忆条数
    max_short_term_history: int = 20  # 短期记忆最大条数（Human+AI 各算一条）

    # === 对话持久化 ===
    chat_history_file: str = "./data/chat_history.json"  # 对话记录保存路径
    
    # === 伴侣人设 ===
    companion_name: str = "小梦"
    companion_personality: str = "温柔、善解人意、有点俏皮"
    companion_backstory: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # 忽略、外字段


# 全局配置实例
settings = Settings()
