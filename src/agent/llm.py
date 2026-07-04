import os
from langchain_openai import ChatOpenAI
from src.config.settings import settings

def get_llm():
    """获取 LLM 实例（DeepSeek 兼容 OpenAI API）"""
    # 清除可能导致 SSL 错误的环境变量
    os.environ.pop("SSL_CERT_FILE", None)
    
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
    )
