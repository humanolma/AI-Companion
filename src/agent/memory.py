"""
长期记忆模块 — 使用 ChromaDB 向量数据库存储和检索对话历史

⚠️ 注意：DeepSeek 不支持 embeddings API，本模块使用本地 sentence-transformers 模型
"""
from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class LocalEmbeddings(Embeddings):
    """本地 sentence-transformers embedding 包装类（不依赖 langchain 版本）"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


class LongTermMemory:
    """长期记忆管理器（基于 ChromaDB 向量数据库）"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db_path = settings.chroma_persist_dir
        self.collection_name = f"memory_{user_id}"

        # 使用本地 sentence-transformers 模型做 embedding（无需外部 API）
        self.embeddings = LocalEmbeddings(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
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
