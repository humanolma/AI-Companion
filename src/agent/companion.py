"""
CompanionAgent — 虚拟伴侣 Agent 主类

功能：
1. 短期记忆（对话上下文，最近 N 轮）
2. 长期记忆（ChromaDB 向量数据库，可选启用）
3. 情感感知（识别用户情绪，动态调整回复语气）
4. 流式输出（打字机效果）
"""
import json
import os
from typing import List, Optional, Generator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent.llm import get_llm
from src.agent.emotion import EmotionDetector
from src.config.settings import settings

class CompanionAgent:
    """虚拟伴侣 Agent 主类"""

    def __init__(self, use_long_term_memory: bool = False, use_emotion: bool = True):
        """
        初始化 Agent
        :param use_long_term_memory: 是否启用长期记忆（ChromaDB）
        :param use_emotion: 是否启用情感感知
        """
        self.llm = get_llm()
        self.system_prompt_template = self._build_system_prompt()
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

        # 短期记忆（对话上下文）
        self.history: List = []

        # 长期记忆（可选）
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory = None
        if self.use_long_term_memory:
            from src.agent.memory import get_memory
            self.long_term_memory = get_memory()

        # 情感感知（可选）
        self.use_emotion = use_emotion
        self.emotion_detector = EmotionDetector() if use_emotion else None
        # 记录最近一次检测到的情绪
        self.current_emotion: str = "neutral"

        # 加载持久化的对话历史
        self._load_history()

    def _build_system_prompt(self) -> str:
        """构建系统 Prompt（静态部分）"""
        base = f"""你是 {settings.companion_name}，一个 {settings.companion_personality} 的 AI 伴侣。

{settings.companion_backstory if settings.companion_backstory else ''}

你的特点：
1. 说话风格：{settings.companion_personality}
2. 会记住和用户的对话历史
3. 会在合适的时候主动关心用户
4. 回复简洁、自然，不要太长
5. 适当使用表情符号增加亲和力

当前对话中，请根据历史对话上下文，给出贴心、自然的回复。
"""
        return base

    def _build_prompt(self) -> ChatPromptTemplate:
        """构建 Prompt 模板（LangChain 格式）"""
        return ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

    def _format_long_term_context(self, memories: List[str]) -> str:
        """将检索到的长期记忆格式化为文本"""
        if not memories:
            return ""
        return "\n【长期记忆】以下是你之前和用户的对话片段，请参考：\n" + "\n---\n".join(memories)

    def _prepare_context(self, user_input: str) -> str:
        """组装系统消息（情绪调整 + 长期记忆），供 chat 和 chat_stream 复用"""
        # 情感感知
        emotion_adjustment = ""
        if self.use_emotion and self.emotion_detector:
            self.current_emotion = self.emotion_detector.detect(user_input)
            emotion_adjustment = self.emotion_detector.get_adjustment(self.current_emotion)

        system_message = self.system_prompt_template
        if emotion_adjustment:
            system_message += f"\n\n【情绪感知】用户当前情绪：{self.current_emotion}\n回复调整：{emotion_adjustment}"
        if self.use_long_term_memory and self.long_term_memory:
            memories = self.long_term_memory.retrieve_memories(user_input)
            if memories:
                system_message += "\n" + self._format_long_term_context(memories)
        return system_message

    def chat(self, user_input: str) -> str:
        """与用户对话（一次性返回完整回复）"""
        if not user_input:
            return ""

        system_message = self._prepare_context(user_input)

        # 调用 LLM
        response = self.chain.invoke({
            "system_message": system_message,
            "history": self.history,
            "input": user_input,
        })

        # 更新短期记忆
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))
        if len(self.history) > settings.max_short_term_history:
            self.history = self.history[-settings.max_short_term_history:]

        # 存储长期记忆
        if self.use_long_term_memory and self.long_term_memory:
            self.long_term_memory.add_memory(user_input, response.content)

        # 持久化对话历史
        self._save_history()

        return response.content

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        流式对话（逐字 yield 回复内容，打字机效果）
        :yield: 每次返回一个文本片段
        """
        if not user_input:
            return

        system_message = self._prepare_context(user_input)

        # 流式调用 LLM，逐 chunk 拼接
        full_response = ""
        for chunk in self.chain.stream({
            "system_message": system_message,
            "history": self.history,
            "input": user_input,
        }):
            piece = chunk.content
            full_response += piece
            yield piece

        # 流结束后更新记忆
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=full_response))
        if len(self.history) > settings.max_short_term_history:
            self.history = self.history[-settings.max_short_term_history:]

        if self.use_long_term_memory and self.long_term_memory:
            self.long_term_memory.add_memory(user_input, full_response)

        # 持久化对话历史
        self._save_history()

    def reset(self):
        """重置对话（只清空短期记忆，不删文件）"""
        self.history = []

    # ========== 对话持久化 ==========

    def _load_history(self):
        """从 JSON 文件加载历史对话"""
        filepath = settings.chat_history_file
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = []
            for item in data:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    self.history.append(HumanMessage(content=content))
                elif role == "assistant":
                    self.history.append(AIMessage(content=content))
        except Exception:
            self.history = []

    def _save_history(self):
        """保存对话历史到 JSON 文件"""
        filepath = settings.chat_history_file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                data.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                data.append({"role": "assistant", "content": msg.content})
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_history_display(self) -> list:
        """返回前端可用的历史对话格式，供 UI 加载"""
        result = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result

    def clear_chat_history(self):
        """清除对话历史（JSON 文件 + 内存）"""
        self.history = []
        filepath = settings.chat_history_file
        if os.path.exists(filepath):
            os.remove(filepath)

    def clear_long_term_memory(self):
        """清除长期记忆（ChromaDB 全部数据）"""
        if self.long_term_memory:
            self.long_term_memory.clear_memories()

    def clear_all_data(self):
        """一键清除所有数据（对话历史 + 长期记忆）"""
        self.clear_chat_history()
        self.clear_long_term_memory()
