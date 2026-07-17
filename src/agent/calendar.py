"""
日程管理 — 自然语言创建事件、JSON 持久化、LangChain Tool、.ics 系统日历导出
"""

import json
import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
ICS_DIR = "./data/calendar_ics"


def _make_ics(event: dict) -> str:
    """生成 .ics 文件内容（带 5 分钟提前提醒）"""
    dt = datetime.fromisoformat(event["datetime"])
    dt_start = dt.strftime("%Y%m%dT%H%M%S")
    dt_end = (dt + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
    uid = event["id"] + "@ai-companion"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AI Companion//CN
BEGIN:VEVENT
DTSTART:{dt_start}
DTEND:{dt_end}
SUMMARY:{event['title']}
DESCRIPTION:{event.get('notes', '')}
UID:{uid}
DTSTAMP:{now}
BEGIN:VALARM
TRIGGER:-PT5M
ACTION:DISPLAY
DESCRIPTION:⏰ {event['title']} 即将开始
END:VALARM
END:VEVENT
END:VCALENDAR"""


class CalendarManager:
    """日程管理器：CRUD + JSON 持久化 + .ics 导出"""

    def __init__(self, data_file: str = "./data/calendar.json"):
        self._file = data_file
        self._events: List[dict] = []
        self._load()

    # ── CRUD ────────────────────────────────────────────────

    def add_event(self, title: str, datetime_str: str, notes: str = "") -> dict:
        """添加日程。datetime_str 应为 ISO 格式如 '2026-07-16T15:00'"""
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
        self._save_ics(event)
        logger.info("日程已创建: %s @ %s", title, datetime_str)
        return event

    def list_events(self, upcoming_only: bool = True) -> List[dict]:
        self._cleanup()  # 每次列出前先清理过期日程
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

    def get_ics_path(self, event_id: str) -> str | None:
        path = os.path.join(ICS_DIR, f"{event_id}.ics")
        return path if os.path.exists(path) else None

    # ── 内部 ────────────────────────────────────────────────

    def _cleanup(self, keep_days: int = 30):
        """自动删除 N 天前的过期日程"""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        before = len(self._events)
        self._events = [e for e in self._events if e["datetime"] > cutoff]
        if len(self._events) < before:
            self._save()
            logger.info("已清理 %d 条过期日程", before - len(self._events))

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

    def _save_ics(self, event: dict):
        """生成 .ics 文件，双击即可导入系统日历（Windows/Outlook/Google Calendar）"""
        os.makedirs(ICS_DIR, exist_ok=True)
        path = os.path.join(ICS_DIR, f"{event['id']}.ics")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_make_ics(event))
        logger.info(".ics 已生成: %s", path)


# ── 全局单例 ────────────────────────────────────────────────

_calendar_manager: CalendarManager | None = None


def get_calendar_manager(data_file: str = "./data/calendar.json") -> CalendarManager:
    global _calendar_manager
    if _calendar_manager is None:
        _calendar_manager = CalendarManager(data_file=data_file)
    return _calendar_manager


def make_calendar_tool(mgr: CalendarManager):
    """LangChain Tool — LLM 可调 schedule_event"""

    @tool
    def schedule_event(title: str, datetime: str, notes: str = "") -> str:
        """创建日程。title=名称，datetime=ISO格式(如2026-07-16T15:00)，notes=备注"""
        evt = mgr.add_event(title, datetime, notes)
        ics = mgr.get_ics_path(evt["id"])
        return (
            f"已创建日程：{evt['title']}，时间 {evt['datetime']}。"
            + (f" .ics 文件已生成，双击导入系统日历即可获得提醒。" if ics else "")
        )

    return schedule_event
