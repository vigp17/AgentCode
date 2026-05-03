"""
AgentCode — MCP (Model Context Protocol) client.

Connects to MCP servers defined in .agentcode/mcp.json and exposes
their tools to the agent loop alongside the built-in tools.

Config format (.agentcode/mcp.json or ~/.agentcode/mcp.json):
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "ghp_..."}
    }
  }
}

Tools are exposed as mcp__<server>__<tool> (e.g. mcp__filesystem__read_file).
"""

import asyncio
import json
import os
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from rich.console import Console

console = Console()


class _ServerConnection:
    """Holds a live stdio connection to one MCP server."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._session = None
        self._stack: AsyncExitStack | None = None

    async def start(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env.update(self.config.get("env") or {})

        params = StdioServerParameters(
            command=self.config["command"],
            args=self.config.get("args", []),
            env=env,
        )
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def stop(self):
        if self._stack:
            await self._stack.aclose()

    async def list_tools(self):
        return await self._session.list_tools()

    async def call_tool(self, name: str, args: dict):
        return await self._session.call_tool(name, args)


class MCPManager:
    """
    Manages connections to one or more MCP servers.

    Runs a persistent asyncio event loop in a background thread so the
    synchronous agent loop can call MCP tools without restructuring.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers: dict[str, _ServerConnection] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}  # mcp__server__tool -> (server, tool)

    def _run(self, coro, timeout: int = 30):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def connect(self, name: str, config: dict) -> None:
        conn = _ServerConnection(name, config)
        self._run(conn.start())
        self._servers[name] = conn
        tools = self._run(conn.list_tools())
        for tool in tools.tools:
            key = f"mcp__{name}__{tool.name}"
            self._tool_map[key] = (name, tool.name)

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_map

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-format tool definitions for all connected MCP tools."""
        defs = []
        for server_name, conn in self._servers.items():
            tools = self._run(conn.list_tools())
            for tool in tools.tools:
                key = f"mcp__{server_name}__{tool.name}"
                schema = tool.inputSchema if hasattr(tool, "inputSchema") else {
                    "type": "object", "properties": {}
                }
                defs.append({
                    "type": "function",
                    "function": {
                        "name": key,
                        "description": f"[{server_name}] {tool.description or tool.name}",
                        "parameters": schema,
                    },
                })
        return defs

    def call_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self._tool_map:
            return f"Error: Unknown MCP tool '{tool_name}'"
        server_name, original_name = self._tool_map[tool_name]
        conn = self._servers[server_name]
        try:
            result = self._run(conn.call_tool(original_name, args))
        except Exception as e:
            return f"Error calling MCP tool '{tool_name}': {e}"

        if not result.content:
            return "Tool returned no content."
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary data: {len(block.data)} bytes]")
        return "\n".join(parts) or "Tool returned no content."

    def server_names(self) -> list[str]:
        return list(self._servers.keys())

    def shutdown(self):
        for conn in self._servers.values():
            try:
                self._run(conn.stop(), timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


# ── Config loading ─────────────────────────────────────────────────────────────

def load_mcp_config(project_dir: str) -> dict:
    config: dict = {}
    for path in [
        Path.home() / ".agentcode" / "mcp.json",
        Path(project_dir) / ".agentcode" / "mcp.json",
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                config.update(data.get("mcpServers", {}))
            except Exception as e:
                console.print(f"[warning]⚠ Failed to parse MCP config {path}: {e}[/warning]")
    return config


def create_mcp_manager(project_dir: str) -> "MCPManager | None":
    """Load config and connect to all configured MCP servers. Returns None if none configured."""
    try:
        import mcp  # noqa: check available
    except ImportError:
        return None

    config = load_mcp_config(project_dir)
    if not config:
        return None

    manager = MCPManager()
    for name, server_config in config.items():
        try:
            manager.connect(name, server_config)
            console.print(f"[dim]  MCP: connected to [bold]{name}[/bold] "
                          f"({len([k for k in manager._tool_map if k.startswith(f'mcp__{name}__')])} tools)[/dim]")
        except Exception as e:
            console.print(f"[warning]⚠ MCP server '{name}' failed: {e}[/warning]")

    return manager if manager._servers else None
