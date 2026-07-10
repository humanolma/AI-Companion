from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置类（从环境变量 + .env 文件读取）"""
    
    # === API 配置 ===
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    dashscope_api_key: str = ""
    amap_maps_api_key: str = ""  # 高德地图 API Key（用于 MCP 地图工具）
    tavily_api_key: str = ""  # Tavily Search API Key（用于 MCP 联网搜索）
    
    # === 模型配置 ===
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.7
    
    # === Embedding 配置（用于长期记忆）===
    # 使用阿里云 DashScope text-embedding-v3
    embedding_model: str = "text-embedding-v3"
    embedding_device: str = "cpu"  # "cpu" or "cuda"（仅本地模型使用）
    
    # === 记忆配置 ===
    chroma_persist_dir: str = "./data/chroma"
    memory_retrieval_k: int = 3  # 每次检索返回的最相关记忆条数
    max_short_term_history: int = 20  # 短期记忆最大条数（Human+AI 各算一条）

    # === 对话持久化 ===
    chat_history_file: str = "./data/chat_history.json"  # 对话记录保存路径
    user_profile_file: str = "./data/user_profile.json"  # 用户画像
    
    # === MCP 工具配置 ===
    # 高德地图 MCP 由 amap_maps_api_key（系统环境变量）自动启用
    # 额外 MCP 服务器可在此 JSON 文件中配置（与 langchain-mcp-adapters 格式一致）
    mcp_servers_json: str = "./data/mcp_servers.json"

    # === 用量监控与成本控制 ===
    daily_budget_limit: float = 0           # 每日预算上限（元），0 = 不限制
    usage_data_file: str = "./data/usage.json"
    deepseek_input_price: float = 0.55       # 输入价格（元/百万 tokens）
    deepseek_output_price: float = 2.19      # 输出价格（元/百万 tokens）
    
    # === 伴侣人设 ===
    companion_name: str = "小梦"
    companion_personality: str = "温柔、善解人意、有点俏皮"
    companion_backstory: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # 忽略、外字段

# 全局配置实例
settings = Settings()
