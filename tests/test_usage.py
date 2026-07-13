"""测试 UsageTracker 用量统计 + 预算控制"""

import os
import pytest
from src.agent.usage import UsageTracker, estimate_tokens


class TestEstimateTokens:
    """Token 估算函数测试"""

    def test_empty_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_chinese_text(self):
        # "你好世界" 4个汉字 ≈ 2.7 tokens → 至少 1
        assert estimate_tokens("你好世界") >= 1

    def test_english_text(self):
        # "hello world" ≈ 3 tokens
        assert estimate_tokens("hello world") >= 2

    def test_returns_int(self):
        assert isinstance(estimate_tokens("任意文本 some text"), int)


class TestUsageTracker:
    """用量追踪器测试"""

    @pytest.fixture
    def tracker_file(self, tmp_path):
        return str(tmp_path / "usage.json")

    def test_record_increments_counts(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file)
        tracker.record(100, 50)
        stats = tracker.get_stats()
        assert stats["tokens_in"] == 100
        assert stats["tokens_out"] == 50
        assert stats["tokens_total"] == 150
        assert stats["request_count"] == 1

    def test_multiple_records_accumulate(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file)
        tracker.record(100, 50)
        tracker.record(200, 100)
        stats = tracker.get_stats()
        assert stats["tokens_total"] == 450
        assert stats["request_count"] == 2

    def test_persists_across_instances(self, tracker_file):
        t1 = UsageTracker(data_file=tracker_file)
        t1.record(100, 50)

        t2 = UsageTracker(data_file=tracker_file)
        stats = t2.get_stats()
        assert stats["tokens_in"] == 100

    def test_budget_unlimited_when_zero(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file, budget=0)
        tracker.record(1000000, 1000000)  # 大量 tokens
        assert tracker.check_budget() is True

    def test_budget_exceeded(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file, budget=0.001)
        tracker.record(1000, 10000)  # 超过 0.001 元
        assert tracker.check_budget() is False

    def test_budget_within_limit(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file, budget=100.0)
        tracker.record(10, 10)
        assert tracker.check_budget() is True

    def test_stats_includes_budget_info(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file, budget=1.0)
        stats = tracker.get_stats()
        assert stats["budget_limit"] == 1.0
        assert stats["budget_remaining"] is not None

    def test_stats_no_budget_when_zero(self, tracker_file):
        tracker = UsageTracker(data_file=tracker_file, budget=0)
        stats = tracker.get_stats()
        assert stats["budget_limit"] is None
