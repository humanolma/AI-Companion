"""
用户画像 — 从对话中自动提取个人信息，沉淀为结构化档案

用法：
    from src.agent.profile import UserProfile

    profile = UserProfile(data_file="./data/user_profile.json")
    profile.extract_and_merge(user_input, ai_response)  # 每轮对话后调用
    context = profile.format_context()                   # 注入 System Prompt
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """你是一个用户信息提取助手。分析以下对话，提取用户透露的个人信息。

请只返回 JSON 格式，不要其他内容。每个字段如无新信息则为 null：

{
  "name": "用户的名字或称呼",
  "age": "年龄或生日",
  "location": "所在城市或地区",
  "occupation": "职业或工作领域",
  "interests": "兴趣爱好，多个用逗号分隔",
  "skills": "技能或正在学习的技术，多个用逗号分隔",
  "status": "当前生活状态（如求职中、在读、自由职业等）",
  "other": "其他值得记住的个人信息"
}

规则：
- 只提取用户明确提到的信息，不要推测
- 如果对话中没有新的个人信息，所有字段返回 null
- 如果用户更正之前的信息，以最新为准"""


class UserProfile:
    """用户画像管理器"""

    def __init__(self, data_file: str = "./data/user_profile.json"):
        self._data_file = data_file
        self._profile: dict = {}
        self._load()

    # ── 公共 API ──────────────────────────────────────────────

    def extract_and_merge(self, user_input: str, ai_response: str, llm) -> bool:
        """分析一轮对话，提取个人信息并合并到画像。
        :return: 是否有新信息被提取
        """
        try:
            messages = [
                {"role": "user", "content": f"用户消息：{user_input}\nAI回复：{ai_response}\n\n{EXTRACTION_PROMPT}"},
            ]
            # 直接调 LLM（走 OpenAI 兼容 API，非流式）
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content="你只返回 JSON，不要其他内容。"),
                HumanMessage(content=f"用户消息：{user_input}\nAI回复：{ai_response}\n\n{EXTRACTION_PROMPT}"),
            ])
            raw = response.content.strip()
            # 去掉可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            data = json.loads(raw)
        except Exception:
            return False

        updated = False
        for key, value in data.items():
            if value and value != self._profile.get(key):
                self._profile[key] = value
                updated = True
                logger.info("画像更新: %s = %s", key, value)

        if updated:
            self._save()
        return updated

    def get_profile(self) -> dict:
        return dict(self._profile)

    def format_context(self) -> str:
        """格式化为 System Prompt 片段"""
        if not self._profile:
            return ""
        lines = ["\n\n【用户画像】以下是你已知的用户个人信息，请在对话中自然引用："]
        labels = {
            "name": "名字", "age": "年龄", "location": "所在地",
            "occupation": "职业", "interests": "兴趣", "skills": "技能",
            "status": "状态", "other": "其他",
        }
        for key, label in labels.items():
            if self._profile.get(key):
                lines.append(f"- {label}：{self._profile[key]}")
        return "\n".join(lines)

    def clear(self):
        self._profile = {}
        if os.path.exists(self._data_file):
            os.remove(self._data_file)

    # ── 内部 ──────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        try:
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(self._profile, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("画像保存失败")

    def _load(self):
        if os.path.exists(self._data_file):
            try:
                with open(self._data_file, "r", encoding="utf-8") as f:
                    self._profile = json.load(f)
            except Exception:
                self._profile = {}
