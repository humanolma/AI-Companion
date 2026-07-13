"""
长期记忆模块 — 使用 ChromaDB 向量数据库存储和检索对话历史

特性：
- DashScope text-embedding-v3 语义检索
- 时间衰减：越久远的记忆权重越低（可配置半衰期）
- 自动持久化
"""

import math
import time
import requests
from datetime import datetime, timezone
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> float:
    """解析 ISO 时间戳为 Unix 时间（秒），失败返回 0"""
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0


class LongTermMemory:
    """长期记忆管理器（基于 ChromaDB + 时间衰减）"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.db_path = settings.chroma_persist_dir
        self.collection_name = f"memory_{user_id}"

        self.embeddings = DashScopeEmbeddings(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
        )

        self.vectorstore = None
        self.retriever = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.memory_retrieval_k}
        )

    # ── 写入 ──────────────────────────────────────────────────

    def add_memory(self, user_input: str, agent_response: str):
        text = f"用户：{user_input}\n伴侣：{agent_response}"
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{
                "user_id": self.user_id,
                "type": "conversation",
                "timestamp": _now_iso(),
            }],
        )

    # ── 检索（含时间衰减）─────────────────────────────────────

    def retrieve_memories(self, query: str) -> List[str]:
        """检索最相关记忆，加入时间衰减后重排序"""
        if not self.vectorstore:
            return []

        half_life = settings.memory_decay_half_life
        now_ts = time.time()

        # 多取一些候选，供衰减后筛选
        fetch_k = max(settings.memory_retrieval_k * 3, 10)
        results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)

        scored = []
        for doc, sim_score in results:
            ts_str = doc.metadata.get("timestamp", "")
            doc_ts = _parse_iso(ts_str) if ts_str else 0

            # 时间衰减：exp(-age_seconds / half_life_seconds)
            age = now_ts - doc_ts if doc_ts > 0 else 0
            if half_life > 0 and age > 0:
                decay = math.exp(-age / (half_life * 86400))
            else:
                decay = 1.0

            # 综合分：语义相似度 * 时间衰减（sim_score 越小越相似）
            combined = (1.0 - sim_score) * decay
            scored.append((combined, doc.page_content))

        # 按综合分降序，取 top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:settings.memory_retrieval_k]]

    def clear_memories(self):
        self.vectorstore.delete_collection()
        self._init_vectorstore()


def get_memory(user_id: str = "default") -> LongTermMemory:
    return LongTermMemory(user_id=user_id)
