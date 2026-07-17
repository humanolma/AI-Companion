"""
日程管理 — 自然语言创建事件、JSON 持久化、LangChain Tool

用法：
    from src.agent.calendar import CalendarManager, make_calendar_tool

    mgr = CalendarManager()
    tool = make_calendar_tool(mgr)
    mgr.add_event("开会", "2026-07-16T15:00", "产品评审")
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class CalendarManager:
    """日程管理器：CRUD + JSON 持久化"""

    def __init__(self, data_file: str = "./data/calendar.json"):
        self._file = data_file
        self._events: List[dict] = []
        self._load()

    # ── CRUD ────────────────────────────────────────────────

    def add_event(self, title: str, datetime_str: str, notes: str = "") -> dict:
        """添加一个日程。datetime_str 应为 ISO 格式如 '2026-07-16T15:00'"""
        event = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "datetime": datetime_str,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._events.append(event)
        self._events.sort(key=lambda e: e["datetime"])
        self._save()
        logger.info("日程已创建: %s @ %s", title, datetime_str)
        return event

    def list_events(self, upcoming_only: bool = True) -> List[dict]:
        """列出日程。upcoming_only=True 只返回未来的"""
        now = datetime.now(timezone.utc).isoformat()
        if upcoming_only:
            return [e for e in self._events if e["datetime"] > now]
        return list(self._events)

    def delete_event(self, event_id: str) -> bool:
        for e in self._events:
            if e["id"] == event_id:
                self._events.remove(e)
                self._save()
                return True
        return False

    # ── 内部 ────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._events, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("日程保存失败")

    def _load(self):
        if os.path.exists(self._file):
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    self._events = json.load(f)
            except Exception:
                self._events = []


# ── 全局单例 ────────────────────────────────────────────────

_calendar_manager: CalendarManager | None = None


def get_calendar_manager(data_file: str = "./data/calendar.json") -> CalendarManager:
    global _calendar_manager
    if _calendar_manager is None:
        _calendar_manager = CalendarManager(data_file=data_file)
    return _calendar_manager


def make_calendar_tool(mgr: CalendarManager):
    """创建 LangChain Tool，让 LLM 可以调 schedule_event"""

    @tool
    def schedule_event(title: str, datetime: str, notes: str = "") -> str:
        """创建日程提醒。title=事件名称，datetime=ISO格式时间（如 2026-07-16T15:00），notes=备注（可选）"""
        evt = mgr.add_event(title, datetime, notes)
        return f"已创建日程：{evt['title']}，时间 {evt['datetime']}"

    return schedule_event
