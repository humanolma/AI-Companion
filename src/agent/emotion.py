"""
情感感知模块 — 识别用户情绪，调整 Agent 回复语气

实现方式：
1. 用 LLM 对用户输入做情绪分析（structured output）
2. 根据情绪标签，动态调整 system prompt 中的回复风格

情绪类型：happy / sad / angry / anxious / neutral
"""
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.llm import get_llm
from src.config.settings import settings

# === 情绪定义 ===
EMOTIONS = {
    "happy": {
        "label": "开心",
        "adjust": "用户现在心情不错，你可以更活泼俏皮一些，分享用户的快乐，语气轻快。",
    },
    "sad": {
        "label": "难过",
        "adjust": "用户似乎心情不太好，请用温柔、安慰的语气，先表达关心，不要急着给建议，多倾听。",
    },
    "angry": {
        "label": "愤怒",
        "adjust": "用户现在有些生气，请保持冷静和同理心，先认同用户的感受，不要反驳，语气平和。",
    },
    "anxious": {
        "label": "焦虑",
        "adjust": "用户似乎有些焦虑，请用安抚、鼓励的语气，帮助用户放松，给出积极的支持。",
    },
    "neutral": {
        "label": "平静",
        "adjust": "保持你一贯的温柔风格，自然地回应用户即可。",
    },
}

# 情绪分析的 Prompt
EMOTION_ANALYSIS_PROMPT = """你是一个情绪分析助手。请分析用户消息的情绪，只返回以下标签之一：

happy, sad, angry, anxious, neutral

规则：
- happy: 用户表达开心、兴奋、满意、感谢等正面情绪
- sad: 用户表达难过、失落、孤独、沮丧等情绪
- angry: 用户表达愤怒、不满、抱怨、烦躁等情绪
- anxious: 用户表达焦虑、担心、紧张、害怕等情绪
- neutral: 情绪不明显，或为中性/平静的陈述

只返回标签单词，不要返回其他任何内容。"""


class EmotionDetector:
    """情绪检测器：用 LLM 分析用户消息的情绪"""

    def __init__(self):
        self.llm = get_llm()
        # 缓存上一次检测结果（避免重复分析）
        self._last_input: Optional[str] = None
        self._last_emotion: str = "neutral"

    def detect(self, user_input: str) -> str:
        """
        分析用户消息的情绪
        :return: 情绪标签（happy/sad/angry/anxious/neutral）
        """
        if not user_input or not user_input.strip():
            return "neutral"

        # 缓存命中
        if user_input == self._last_input:
            return self._last_emotion

        try:
            response = self.llm.invoke([
                SystemMessage(content=EMOTION_ANALYSIS_PROMPT),
                HumanMessage(content=user_input),
            ])
            emotion = response.content.strip().lower()

            # 校验返回值
            if emotion not in EMOTIONS:
                emotion = "neutral"

            # 更新缓存
            self._last_input = user_input
            self._last_emotion = emotion
            return emotion

        except Exception:
            # LLM 调用失败时降级为 neutral
            return "neutral"

    def get_adjustment(self, emotion: str) -> str:
        """
        根据情绪标签获取回复风格调整指令
        :return: 要追加到 system prompt 的指令文本
        """
        return EMOTIONS.get(emotion, EMOTIONS["neutral"])["adjust"]

    def get_label(self, emotion: str) -> str:
        """获取情绪的中文标签"""
        return EMOTIONS.get(emotion, EMOTIONS["neutral"])["label"]
