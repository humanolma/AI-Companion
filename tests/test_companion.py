"""测试 CompanionAgent 核心链路：对话、记忆、上下文"""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch
from src.agent.companion import CompanionAgent
from src.agent.usage import UsageTracker


def _make_mock_llm(response_text="你好！我是小梦～"):
    """创建一个行为像 LangChain ChatOpenAI 的 mock"""
    mock = MagicMock()
    mock.invoke.return_value.content = response_text
    mock.stream.return_value = iter([_Chunk(response_text)])
    mock.bind_tools.return_value = mock
    return mock


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class TestBasicChat:
    """对话生成测试"""

    def test_chat_updates_history(self, temp_settings):
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
            agent.chat("第一条消息")
        assert len(agent.history) == 2

    def test_reset_clears_history(self, temp_settings):
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
            agent.chat("测试")
        assert len(agent.history) > 0
        agent.reset()
        assert len(agent.history) == 0

    def test_empty_input_returns_empty(self, temp_settings):
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
            assert agent.chat("") == ""
            assert agent.chat("   ") == ""

    def test_budget_exceeded_blocks_chat(self, temp_settings):
        tracker = UsageTracker(budget=0.001)
        tracker.record(500, 500)
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False,
                                   usage_tracker=tracker)
            chunks = list(agent.chat_stream("你好"))
        assert "预算" in chunks[0]


class TestSystemPrompt:
    """System Prompt 组装测试"""

    def test_prompt_contains_companion_name(self, temp_settings):
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            from src.config.settings import settings
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
            prompt = agent._build_system_prompt()
            assert settings.companion_name in prompt

    def test_prompt_contains_personality(self, temp_settings):
        with patch("src.agent.llm.ChatOpenAI", return_value=_make_mock_llm()):
            from src.config.settings import settings
            agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
            prompt = agent._build_system_prompt()
            assert settings.companion_personality in prompt

    def test_prepare_context_adds_emotion_when_enabled(self, mock_emotion_happy, temp_settings):
        agent = CompanionAgent(use_long_term_memory=False, use_emotion=True)
        ctx = agent._prepare_context("我好开心呀")
        assert ("开心" in ctx or "happy" in ctx.lower())
        assert agent.current_emotion == "happy"

    def test_prepare_context_no_emotion_when_disabled(self, mock_emotion_happy, temp_settings):
        agent = CompanionAgent(use_long_term_memory=False, use_emotion=False)
        ctx = agent._prepare_context("我好开心呀")
        assert "情绪感知" not in ctx
