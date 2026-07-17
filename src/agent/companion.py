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
import logging
import httpx
from collections import defaultdict
from datetime import date, datetime
from typing import Generator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent.llm import get_llm
from src.agent.emotion import EmotionDetector
from src.agent.usage import estimate_tokens
from src.agent.profile import UserProfile
from src.agent.calendar import get_calendar_manager, make_calendar_tool
from src.config.settings import settings

logger = logging.getLogger(__name__)

class CompanionAgent:
    """虚拟伴侣 Agent 主类"""

    def __init__(self, use_long_term_memory: bool = False, use_emotion: bool = True,
                 tools: list = None, usage_tracker=None):
        """
        初始化 Agent
        :param use_long_term_memory: 是否启用长期记忆（ChromaDB）
        :param use_emotion: 是否启用情感感知
        :param tools: MCP 工具列表（可后续通过 set_tools 注入）
        :param usage_tracker: UsageTracker 实例（可选，用于用量监控）
        """
        self.llm = get_llm()
        self.system_prompt_template = self._build_system_prompt()
        self.prompt = self._build_prompt()
        self.chain = self.prompt | self.llm

        # 用量追踪（可选）
        self.usage_tracker = usage_tracker

        # 用户画像（从对话中自动提取个人信息）
        self.user_profile = UserProfile(data_file=settings.user_profile_file)

        # 日程管理
        self.calendar = get_calendar_manager()

        # MCP 工具（运行时可由 server lifespan 注入）
        self._bind_tools(tools)

        # 短期记忆（对话上下文）
        self.history: list = []

        # 长期记忆（可选）
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory = None
        if self.use_long_term_memory:
            from src.agent.memory import get_memory
            self.long_term_memory = get_memory()

        # 情感感知（可选）
        self.use_emotion = use_emotion
        self.emotion_detector = EmotionDetector(usage_tracker=usage_tracker) if use_emotion else None
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
        """构建 Prompt 模板（LangChain 格式）—— 用于无工具的简单链路"""
        return ChatPromptTemplate.from_messages([
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

    def _bind_tools(self, tools):
        """绑定工具到 LLM（MCP + 日程），支持运行时重复注入"""
        self.tools = list(tools) if tools else []
        # 加入日程工具
        if hasattr(self, 'calendar') and self.calendar:
            cal_tool = make_calendar_tool(self.calendar)
            self.tools.append(cal_tool)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

    def set_tools(self, tools):
        """运行时注入工具（由 server 的 lifespan 调用）"""
        self._bind_tools(tools)

    def _build_messages(self, system_message: str, user_input: str) -> list:
        """组装发送给 LLM 的消息列表（系统提示 + 历史 + 当前输入）"""
        messages = [SystemMessage(content=system_message)]
        messages.extend(self.history)  # 历史已是 HumanMessage / AIMessage 列表
        messages.append(HumanMessage(content=user_input))
        return messages

    def _run_with_tools(self, messages: list) -> tuple:
        """带工具调用的 LLM 调用：自动执行工具直到无 tool_calls。
        :return: (最终文本, 累计输入tokens, 累计输出tokens)
        """
        tokens_in = 0
        tokens_out = 0
        # 估算首轮输入
        tokens_in += estimate_tokens("".join(m.content for m in messages))
        response = self.llm_with_tools.invoke(messages)
        tokens_out += estimate_tokens(str(response.content))
        if response.tool_calls:
            tokens_out += estimate_tokens(str(response.tool_calls))
        iterations = 0
        while response.tool_calls and iterations < 5:
            messages.append(response)
            for tc in response.tool_calls:
                tool = self.tools_by_name.get(tc["name"])
                if tool:
                    try:
                        result = tool.invoke(tc["args"])
                    except Exception as e:
                        result = f"工具执行失败：{e}"
                else:
                    result = f"未找到工具：{tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            # 估算后续轮次输入（含工具结果）
            tokens_in += estimate_tokens("".join(
                m.content for m in messages[-(1 + len(response.tool_calls)):]
            ))
            response = self.llm_with_tools.invoke(messages)
            tokens_out += estimate_tokens(str(response.content))
            if response.tool_calls:
                tokens_out += estimate_tokens(str(response.tool_calls))
            iterations += 1
        return response.content, tokens_in, tokens_out

    def _format_long_term_context(self, memories: list[str]) -> str:
        """将检索到的长期记忆格式化为文本"""
        if not memories:
            return ""
        return "\n【长期记忆】以下是你之前和用户的对话片段，请参考：\n" + "\n---\n".join(memories)

    def _reverse_geocode(self, location: dict) -> str | None:
        """用 AMap REST API 逆地理编码，返回城市名（同步 HTTP，不依赖 MCP）"""
        try:
            lng = location.get("lng") or location.get("longitude")
            lat = location.get("lat") or location.get("latitude")
            if not lng or not lat:
                return None
            resp = httpx.get("https://restapi.amap.com/v3/geocode/regeo", params={
                "location": f"{lng},{lat}",
                "key": settings.amap_maps_api_key,
                "extensions": "base",
            }, timeout=5.0)
            data = resp.json()
            if data.get("status") == "1":
                regeo = data.get("regeocode", {})
                addr = regeo.get("addressComponent", {})
                city = addr.get("city") or addr.get("province")
                return city if city else None
        except Exception:
            pass
        return None

    def _prepare_context(self, user_input: str, location: dict = None) -> str:
        """组装系统消息（情绪调整 + 长期记忆），供 chat 和 chat_stream 复用"""
        # 情感感知
        emotion_adjustment = ""
        if self.use_emotion and self.emotion_detector:
            self.current_emotion = self.emotion_detector.detect(user_input)
            emotion_adjustment = self.emotion_detector.get_adjustment(self.current_emotion)
            # 记录情绪历史
            self._record_emotion(self.current_emotion)

        system_message = self.system_prompt_template
        if emotion_adjustment:
            system_message += f"\n\n【情绪感知】用户当前情绪：{self.current_emotion}\n回复调整：{emotion_adjustment}"
        if self.use_long_term_memory and self.long_term_memory:
            memories = self.long_term_memory.retrieve_memories(user_input)
            if memories:
                system_message += "\n" + self._format_long_term_context(memories)
        # 工具感知：明确告知可用工具，提升主动调用概率
        if self.tools:
            names = "、".join(getattr(t, "name", str(t)) for t in self.tools)
            system_message += (
                f"\n\n【可用工具】你拥有以下工具来更好地帮助用户：{names}。"
                "当用户需求涉及这些工具的能力（如查地点、路线、周边搜索等）时，主动调用它们，"
                "并把结果自然地融入你的回复中。"
            )
        # 用户画像：注入已知个人信息
        profile_context = self.user_profile.format_context()
        if profile_context:
            system_message += profile_context

        # 已有日程：注入未来日程列表
        upcoming = self.calendar.list_events(upcoming_only=True)
        if upcoming:
            lines = ["\n\n【日程提醒】用户已有的日程安排："]
            for evt in upcoming[:5]:
                lines.append(f"- {evt['title']}（{evt['datetime']}）")
            system_message += "\n".join(lines)

        # 位置感知：逆地理编码获取城市名，注入 System Prompt
        if location and settings.amap_maps_api_key:
            city = self._reverse_geocode(location)
            if city:
                system_message += (
                    f"\n\n【位置感知】用户当前所在城市：{city}。"
                    "当用户询问天气、周边、路线等与位置相关的问题时，优先使用这个城市作为默认位置，无需再问用户在哪。"
                )
        return system_message

    def chat(self, user_input: str, location: dict = None) -> str:
        """与用户对话（一次性返回完整回复）"""
        if not user_input or not user_input.strip():
            return ""

        system_message = self._prepare_context(user_input, location=location)

        # 调用 LLM（有工具时走工具调用循环）
        if self.tools:
            messages = self._build_messages(system_message, user_input)
            response_text, _, _ = self._run_with_tools(messages)
        else:
            response = self.chain.invoke({
                "system_message": system_message,
                "history": self.history,
                "input": user_input,
            })
            response_text = response.content

        # 更新短期记忆
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response_text))
        if len(self.history) > settings.max_short_term_history:
            self.history = self.history[-settings.max_short_term_history:]

        # 存储长期记忆
        if self.use_long_term_memory and self.long_term_memory:
            self.long_term_memory.add_memory(user_input, response_text)

        # 持久化对话历史
        self._save_history()

        # 用户画像提取（静默，失败不影响对话）
        try:
            self.user_profile.extract_and_merge(user_input, response_text, self.llm)
        except Exception:
            pass

        return response_text

    def _estimate_input_tokens(self, system_message: str, user_input: str) -> int:
        """估算本次 LLM 调用的输入 token 数"""
        history_text = "".join(m.content for m in self.history)
        return estimate_tokens(system_message + history_text + user_input)

    def chat_stream(self, user_input: str, location: dict = None) -> Generator[str, None, None]:
        """
        流式对话（逐字 yield 回复内容，打字机效果）
        :param location: 可选，前端传来的 GPS 坐标 {lat, lng}
        :yield: 每次返回一个文本片段
        """
        if not user_input or not user_input.strip():
            return

        system_message = self._prepare_context(user_input, location=location)

        # 预算检查
        if self.usage_tracker and not self.usage_tracker.check_budget():
            yield "⚠️ 今日用量已达预算上限，请明天再试或调整预算设置。"
            return

        tokens_in = self._estimate_input_tokens(system_message, user_input)

        try:
            # 有工具时：先同步完成工具调用（含多轮），再一次性返回最终回复
            if self.tools:
                messages = self._build_messages(system_message, user_input)
                full_response, tool_tokens_in, tool_tokens_out = self._run_with_tools(messages)
                tokens_in += tool_tokens_in
                tokens_out = tool_tokens_out
                yield full_response
            else:
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
                tokens_out = estimate_tokens(full_response)

            # 记录用量
            if self.usage_tracker:
                self.usage_tracker.record(tokens_in, tokens_out)
                logger.debug("用量: +%d in / +%d out tokens", tokens_in, tokens_out)

        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            full_response = "抱歉，我暂时无法回复，请稍后再试～"
            yield full_response
            tokens_out = 0

        # 流结束后更新记忆
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=full_response))
        if len(self.history) > settings.max_short_term_history:
            self.history = self.history[-settings.max_short_term_history:]

        if self.use_long_term_memory and self.long_term_memory:
            self.long_term_memory.add_memory(user_input, full_response)

        # 持久化对话历史
        self._save_history()

        # 用户画像提取（静默，失败不影响对话）
        try:
            self.user_profile.extract_and_merge(user_input, full_response, self.llm)
        except Exception:
            pass

    # ── 情绪历史 ─────────────────────────────────────────

    def _record_emotion(self, emotion: str):
        filepath = "./data/emotion_history.json"
        today = str(date.today())
        now = datetime.now().strftime("%H:%M")
        entry = {"date": today, "time": now, "emotion": emotion}
        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        data.append(entry)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_emotion_history(self, days: int = 7) -> list:
        filepath = "./data/emotion_history.json"
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        # 按天聚合：{date: {happy: N, sad: N, ...}}
        from collections import defaultdict
        cutoff = date.today()
        result = defaultdict(lambda: {"happy": 0, "sad": 0, "angry": 0, "anxious": 0, "neutral": 0})
        for e in data:
            d = e["date"]
            em = e.get("emotion", "neutral")
            if em in result[d]:
                result[d][em] += 1
        return [{"date": d, **counts} for d, counts in sorted(result.items())[-days:]]

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
        """一键清除所有数据（对话历史 + 长期记忆 + 用户画像）"""
        self.clear_chat_history()
        self.clear_long_term_memory()
        self.user_profile.clear()
