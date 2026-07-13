"""
MCP 工具管理器 — 连接 MCP 服务器并加载 LangChain 工具

当前支持：
- 高德地图（AMap）MCP Server：基于 amap_maps_api_key（系统环境变量）自动启用
- 额外 MCP 服务器：从 mcp_servers_json（默认 ./data/mcp_servers.json）读取，可选

实现说明：
- 直接基于官方 mcp SDK（stdio_client + ClientSession）加载工具并封装为 LangChain Tool，
  避免 langchain-mcp-adapters 与 mcp 新版本之间的解析兼容问题。
- 连接失败不影响主服务（降级为无工具运行）。
- 工具调用通过 run_coroutine_threadsafe 回到原始事件循环执行，避免跨循环死锁。
"""
import os
import json
import logging
import asyncio
import subprocess
from typing import List
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.config.settings import settings

logger = logging.getLogger(__name__)


class _MCPTool(BaseTool):
    """把单个 MCP 工具封装成 LangChain Tool（延迟到调用时才真正执行）"""

    # LangChain 要求的字段
    name: str
    description: str
    args_schema: type  # pydantic model

    def __init__(self, tool_spec, session: ClientSession, session_cm, proc_cm, loop: asyncio.AbstractEventLoop):
        from pydantic import create_model
        # 根据 JSON Schema 生成 args_schema
        props = (tool_spec.inputSchema or {}).get("properties", {})
        required = (tool_spec.inputSchema or {}).get("required", [])
        fields = {}
        for pname, pspec in props.items():
            ptype = {
                "string": str, "integer": int, "number": float, "boolean": bool,
            }.get(pspec.get("type"), str)
            default = ... if pname in required else None
            fields[pname] = (ptype, default)
        schema_cls = create_model(f"{tool_spec.name}_args", **fields)

        super().__init__(
            name=tool_spec.name,
            description=tool_spec.description or "",
            args_schema=schema_cls,
        )
        # 存储运行时引用（不进入 pydantic 字段）
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_session_cm", session_cm)
        object.__setattr__(self, "_proc_cm", proc_cm)
        object.__setattr__(self, "_loop", loop)

    def _run(self, **kwargs) -> str:
        """同步包装：通过 run_coroutine_threadsafe 回到原始事件循环执行，避免跨循环死锁"""
        loop = object.__getattribute__(self, "_loop")
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._arun(**kwargs), loop)
            return future.result(timeout=30)
        else:
            # 降级：循环已停止时用新循环（仅在 shutdown 等极端场景）
            return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        result = await self._session.call_tool(self.name, kwargs)
        # 拼接文本内容返回
        parts = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(item.text)
        return "\n".join(parts) if parts else str(result)


class MCPToolManager:
    """管理一个或多个 MCP 服务器的连接，并暴露 LangChain 工具"""

    def __init__(self):
        self._sessions: List[tuple] = []  # (session, session_cm, proc_cm)
        self.tools: List[BaseTool] = []
        self._loop: asyncio.AbstractEventLoop = None
        self._retry_counts: dict = {}  # 重连计数
        self._max_retries = 3

    def _build_server_configs(self) -> dict:
        """构建 MCP 服务器配置（名称 -> 连接参数）"""
        configs: dict = {}

        # 高德地图 MCP：配置了 API Key 才自动启用
        if settings.amap_maps_api_key:
            configs["amap"] = {
                "command": "npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": settings.amap_maps_api_key,
                },
            }

        # 时间 MCP：无需 API Key，始终启用
        configs["time"] = {
            "command": "python",
            "args": ["-m", "mcp_server_time"],
        }

        # Tavily 搜索 MCP：配置了 API Key 才自动启用
        if settings.tavily_api_key:
            configs["tavily"] = {
                "command": "npx",
                "args": ["-y", "tavily-mcp"],
                "env": {
                    "TAVILY_API_KEY": settings.tavily_api_key,
                },
            }

        # 从 JSON 文件读取额外服务器（同格式：name -> {command, args, env}）
        # 仅含下划线开头的键会被忽略，便于在 JSON 里写注释。
        json_path = settings.mcp_servers_json
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if not k.startswith("_") and isinstance(v, dict):
                            configs[k] = v
            except Exception:
                pass

        return configs

    async def connect(self):
        """连接所有 MCP 服务器并加载工具（幂等：已连接则跳过）"""
        if self._sessions:
            return
        self._loop = asyncio.get_running_loop()  # 保存事件循环，供工具调用
        configs = self._build_server_configs()
        if not configs:
            return
        for name, cfg in configs.items():
            proc_cm = None
            session_cm = None
            try:
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env={**os.environ, **(cfg.get("env") or {})},
                )
                proc_cm = stdio_client(params)
                read_tx, write_tx = await proc_cm.__aenter__()
                session_cm = ClientSession(read_tx, write_tx)
                session = await session_cm.__aenter__()
                await session.initialize()
                self._sessions.append((session, session_cm, proc_cm))
                tools_result = await session.list_tools()
                for tool_spec in tools_result.tools:
                    self.tools.append(_MCPTool(tool_spec, session, session_cm, proc_cm, self._loop))
                logger.info("[MCP] %s 已连接，加载 %d 个工具", name, len(tools_result.tools))
            except Exception as e:
                logger.warning("[MCP] 服务器 %s 连接失败，已跳过：%s", name, e)
                # 清理已进入但未加入 _sessions 的上下文管理器，
                # 否则关闭时它们会成为孤儿，触发 anyio cancel scope 报错
                if session_cm is not None:
                    try:
                        await session_cm.__aexit__(None, None, None)
                    except BaseException:
                        pass
                if proc_cm is not None:
                    try:
                        await proc_cm.__aexit__(None, None, None)
                    except BaseException:
                        pass
        if not self.tools:
            logger.warning("[MCP] 未加载到任何工具，降级为无工具模式")

    def get_tools(self) -> List[BaseTool]:
        return self.tools

    async def disconnect(self):
        """断开 MCP 服务器（释放子进程）"""
        # 先摘除工具，防止关闭期间被 LLM 调用
        self.tools = []
        sessions = self._sessions
        self._sessions = []
        self._loop = None
        for session, session_cm, proc_cm in sessions:
            try:
                await session_cm.__aexit__(None, None, None)
            except BaseException:
                pass
            try:
                await proc_cm.__aexit__(None, None, None)
            except BaseException:
                pass
        sessions.clear()
        # Windows：强制终止残留的 npx / node 子进程，避免 Event loop is closed
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/FI", "IMAGENAME eq node.exe"],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
            # 给 ProactorEventLoop IOCP 回调足够时间处理管道关闭
            for _ in range(3):
                try:
                    await asyncio.sleep(0.15)
                except BaseException:
                    break
