"""
用量追踪 — Token 统计 + 成本估算 + 预算控制

用法：
    from src.agent.usage import UsageTracker, estimate_tokens

    tracker = UsageTracker(data_file="./data/usage.json", budget=1.0)
    tokens_in = estimate_tokens(prompt)
    tokens_out = estimate_tokens(response)
    tracker.record(tokens_in, tokens_out)
    stats = tracker.get_stats()
"""
import json
import os
import re
import logging
from datetime import date

logger = logging.getLogger(__name__)

# DeepSeek v4-flash 默认定价（元/百万 tokens）
DEFAULT_INPUT_PRICE = 0.55
DEFAULT_OUTPUT_PRICE = 2.19


def estimate_tokens(text: str) -> int:
    """字符级 Token 估算：中文 ~1.5 字/token，英文/符号 ~3.5 字/token"""
    if not text:
        return 0
    chinese = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
    other = len(text) - chinese
    return max(1, int(chinese / 1.5 + other / 3.5))


class UsageTracker:
    """用量追踪器：记录 Token 消耗，持久化到 JSON，支持预算控制"""

    def __init__(
        self,
        data_file: str = "./data/usage.json",
        budget: float = 0,
        input_price: float = None,
        output_price: float = None,
    ):
        self._data_file = data_file
        self._budget = budget
        self._input_price = input_price if input_price is not None else DEFAULT_INPUT_PRICE
        self._output_price = output_price if output_price is not None else DEFAULT_OUTPUT_PRICE

        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.request_count: int = 0
        self._today: str = str(date.today())

        self._load()

    # ── 公共 API ──────────────────────────────────────────────

    def record(self, tokens_in: int, tokens_out: int):
        """记录一次 LLM 调用"""
        self._check_day()
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.request_count += 1
        self._save()

    def check_budget(self) -> bool:
        """检查是否超预算，超限返回 False"""
        if self._budget <= 0:
            return True
        self._check_day()
        cost = self._calc_cost(self.tokens_in, self.tokens_out)
        return cost < self._budget

    def get_stats(self) -> dict:
        """返回今日用量统计"""
        self._check_day()
        cost = self._calc_cost(self.tokens_in, self.tokens_out)
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "cost": round(cost, 4),
            "request_count": self.request_count,
            "budget_limit": self._budget if self._budget > 0 else None,
            "budget_remaining": round(max(0, self._budget - cost), 4) if self._budget > 0 else None,
        }

    # ── 内部 ──────────────────────────────────────────────────

    def _calc_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1_000_000) * self._input_price + (
            tokens_out / 1_000_000
        ) * self._output_price

    def _check_day(self):
        today = str(date.today())
        if today != self._today:
            self.tokens_in = 0
            self.tokens_out = 0
            self.request_count = 0
            self._today = today

    def _save(self):
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        data = {
            "date": self._today,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "request_count": self.request_count,
            "budget": self._budget,
            "prices": {
                "input": self._input_price,
                "output": self._output_price,
            },
        }
        try:
            with open(self._data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("用量数据保存失败")

    def _load(self):
        if not os.path.exists(self._data_file):
            return
        try:
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_date = data.get("date", "")
            if saved_date == self._today:
                self.tokens_in = data.get("tokens_in", 0)
                self.tokens_out = data.get("tokens_out", 0)
                self.request_count = data.get("request_count", 0)
            # 日期不同则从零开始（已在 __init__ 中置零）
        except Exception:
            logger.warning("用量数据加载失败，从零开始")
