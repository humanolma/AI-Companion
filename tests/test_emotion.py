"""测试 EmotionDetector 情绪识别链路"""

import pytest
from src.agent.emotion import EmotionDetector, EMOTIONS, EMOTION_ANALYSIS_PROMPT


class TestEmotionDetection:
    """情绪检测测试"""

    def test_detect_happy(self, mock_emotion_happy):
        detector = EmotionDetector()
        result = detector.detect("我今天超级开心！")
        assert result == "happy"

    def test_detect_sad(self, mock_emotion_sad):
        detector = EmotionDetector()
        result = detector.detect("心情不太好")
        assert result == "sad"

    def test_empty_input_returns_neutral(self, mock_emotion_happy):
        detector = EmotionDetector()
        assert detector.detect("") == "neutral"
        assert detector.detect("   ") == "neutral"

    def test_cache_hit_returns_cached(self, mock_emotion_happy):
        detector = EmotionDetector()
        first = detector.detect("你好")
        second = detector.detect("你好")  # 缓存命中
        assert first == second
        # 缓存命中时不调 LLM（如果打了两次则 mock 会报错，这里验证不报错即可）

    def test_get_adjustment_returns_text_for_all_emotions(self):
        detector = EmotionDetector()
        for emotion in EMOTIONS:
            text = detector.get_adjustment(emotion)
            assert len(text) > 0

    def test_all_emotions_have_label(self):
        """每种情绪都有中文标签"""
        detector = EmotionDetector()
        for emotion in EMOTIONS:
            label = detector.get_label(emotion)
            assert len(label) > 0
            assert label != emotion  # 是中文，不是英文标签

    def test_analysis_prompt_not_empty(self):
        """情绪分析 Prompt 非空"""
        assert len(EMOTION_ANALYSIS_PROMPT) > 0
