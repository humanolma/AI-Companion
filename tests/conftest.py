"""pytest fixtures — Mock LLM 避免真实 API 调用"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeChunk:
    def __init__(self, content: str):
        self.content = content


class FakeResponse:
    def __init__(self, content: str, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI，所有调用返回预设文本"""
    with patch("src.agent.llm.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.invoke.return_value = FakeResponse("你好！我是小梦～有什么我可以帮你的吗？")
        instance.stream.return_value = iter([
            FakeChunk("你好！"),
            FakeChunk("我是小梦～"),
            FakeChunk("有什么我可以帮你的吗？"),
        ])
        yield instance


@pytest.fixture
def mock_emotion_happy():
    with patch("src.agent.llm.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.invoke.return_value = FakeResponse("happy")
        yield instance


@pytest.fixture
def mock_emotion_sad():
    with patch("src.agent.llm.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.invoke.return_value = FakeResponse("sad")
        yield instance


@pytest.fixture
def temp_settings():
    """临时覆盖数据路径，避免污染真实数据"""
    import tempfile
    import shutil
    from src.config import settings as s

    tmp = tempfile.mkdtemp()
    # 保存原始值
    _orig = {k: getattr(s.settings, k) for k in [
        "chroma_persist_dir", "chat_history_file", "user_profile_file",
        "usage_data_file", "amap_maps_api_key", "tavily_api_key",
    ]}
    # 覆盖
    s.settings.chroma_persist_dir = os.path.join(tmp, "chroma")
    s.settings.chat_history_file = os.path.join(tmp, "chat.json")
    s.settings.user_profile_file = os.path.join(tmp, "profile.json")
    s.settings.usage_data_file = os.path.join(tmp, "usage.json")
    s.settings.amap_maps_api_key = ""
    s.settings.tavily_api_key = ""

    yield tmp

    # 还原
    for k, v in _orig.items():
        setattr(s.settings, k, v)
    shutil.rmtree(tmp, ignore_errors=True)
