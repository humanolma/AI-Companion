"""
长期记忆模块 — 使用 ChromaDB 向量数据库存储和检索对话历史

⚠️ 注意：使用阿里云 DashScope text-embedding-v3 原生 API
"""
import requests
from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from src.config.settings import settings


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope text-embedding-v3"""

    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def _call_api(self, texts: List[str], text_type: str = "document") -> List[List[float]]:
        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": {"texts": texts},
                "parameters": {"text_type": text_type},
            },
        )
        data = resp.json()
        if data.get("code") and data["code"] != "":
            raise RuntimeError(f"DashScope embedding error: {data}")
        return [e["embedding"] for e in data["output"]["embeddings"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text], text_type="query")[0]


class LongTermMemory:
    """长期记忆管理器（基于 ChromaDB 向量数据库）"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db_path = settings.chroma_persist_dir
        self.collection_name = f"memory_{user_id}"

        # 使用阿里云 DashScope text-embedding-v3 原生 API
        self.embeddings = DashScopeEmbeddings(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
        )

        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        """初始化向量数据库"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.memory_retrieval_k}
        )

    def add_memory(self, user_input: str, agent_response: str):
        """
        将一轮对话存入长期记忆
        """
        # 拼接成一段文本
        text = f"用户：{user_input}\n伴侣：{agent_response}"

        # 存入向量数据库
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{
                "user_id": self.user_id,
                "type": "conversation",
            }],
        )
        # Chroma 会自动持久化（persist_directory 模式下）

    def retrieve_memories(self, query: str) -> List[str]:
        """
        根据用户输入，检索最相关的历史记忆
        返回：相关记忆文本列表（最多 k 条）
        """
        if not self.retriever:
            return []
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def clear_memories(self):
        """清空该用户的长期记忆"""
        self.vectorstore.delete_collection()
        self._init_vectorstore()

def get_memory(user_id: str = "default") -> LongTermMemory:
    """工厂函数：获取长期记忆实例"""
    return LongTermMemory(user_id=user_id)
