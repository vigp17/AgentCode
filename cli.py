#!/usr/bin/env python3
"""
AgentCode — an open, multi-model agentic coding CLI.

Usage:
    agentcode                    # Interactive REPL
    agentcode "fix the bug"      # One-shot mode
    agentcode --model gpt-4o     # Use a different model
    agentcode --no-route         # Disable cost-aware routing

Supported models (via LiteLLM):
    claude-sonnet-4-6            # Anthropic Claude (default)
    gpt-4o                       # OpenAI
    gemini/gemini-2.5-pro        # Google
    ... and 100+ more: https://docs.litellm.ai/docs/providers
"""

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

from agent import (
    AgentConfig, Conversation, run_agent_loop,
    load_project_config, build_system_prompt,
)
from router import ModelRouter
from mcp_client import create_mcp_manager
from settings import (
    load_settings, generate_starter_settings, settings_sources, SETTINGS_HELP,
)
from tools import configure_limits

# ── Theme ─────────────────────────────────────────────────────────────────────

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "prompt": "bold magenta",
})

console = Console(theme=custom_theme)

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
   ___                    __  ______          __
  / _ | ___ ____ ___  ___/ /_/ ___/__  ___/ /__
 / __ |/ _ `/ -_) _ \/ __/ __/ /__/ _ \/ _  / -_)
/_/ |_|\_, /\__/_//_/\__/\__/\___/\___/\_,_/\__/
      /___/
"""


def show_banner(config: AgentConfig, project_config: dict | None = None):
    console.print(Text(BANNER, style="bold cyan"))
    console.print(f"  [dim]Model:[/dim]  [bold]{config.model}[/bold]")
    console.print(f"  [dim]Dir:[/dim]    [bold]{config.project_dir}[/bold]")
    console.print(f"  [dim]Safety:[/dim] [bold]{'auto-approve' if config.auto_approve else 'ask before write/exec'}[/bold]")

    if config.router:
        status = "[bold green]on[/bold green]" if config.router.enabled else "[dim]off[/dim]"
        console.print(f"  [dim]Routing:[/dim][bold] {status}[/bold] [dim](light/medium/heavy tiers)[/dim]")
    else:
        console.print("  [dim]Routing:[/dim] [dim]off[/dim]")

    if config.mcp_manager:
        servers = ", ".join(config.mcp_manager.server_names())
        console.print(f"  [dim]MCP:[/dim]     [bold green]{servers}[/bold green]")

    if project_config:
        if project_config.get("project_path"):
            console.print(f"  [dim]Config:[/dim] [bold green]✓ {project_config['project_path']}[/bold green]")
        if project_config.get("global_path"):
            console.print(f"  [dim]Global:[/dim] [bold green]✓ {project_config['global_path']}[/bold green]")
        if not project_config.get("project_path") and not project_config.get("global_path"):
            console.print("  [dim]Config:[/dim] [dim]no AGENTCODE.md found (optional)[/dim]")

    sources = settings_sources(config.project_dir)
    if sources["project"]:
        console.print(f"  [dim]Settings:[/dim][bold green] ✓ {sources['project']}[/bold green]")
    elif sources["global"]:
        console.print(f"  [dim]Settings:[/dim][bold green] ✓ {sources['global']} (global)[/bold green]")

    console.print()
    console.print("  [dim]Type your request, or:[/dim]")
    console.print("  [dim]  /model <name>  — switch model (disables routing)[/dim]")
    console.print("  [dim]  /route         — show/toggle routing[/dim]")
    console.print("  [dim]  /cost          — show session cost breakdown[/dim]")
    console.print("  [dim]  /mcp           — manage MCP server connections[/dim]")
    console.print("  [dim]  /clear         — reset conversation[/dim]")
    console.print("  [dim]  /compact       — force context compaction[/dim]")
    console.print("  [dim]  /init          — create AGENTCODE.md template[/dim]")
    console.print("  [dim]  /settings      — show resolved settings[/dim]")
    console.print("  [dim]  /tokens        — show token usage[/dim]")
    console.print("  [dim]  /help          — show this help[/dim]")
    console.print("  [dim]  /exit          — quit[/dim]")
    console.print()


# ── Cost display ──────────────────────────────────────────────────────────────

def display_cost_line(config: AgentConfig):
    """Print per-turn and session cost after each response."""
    if not config.router or not config.router.cost_tracker.requests:
        return
    tracker = config.router.cost_tracker
    console.print(
        f"  [dim]💰 ${tracker.last_turn_cost:.4f} "
        f"({tracker.last_turn_input:,} in / {tracker.last_turn_output:,} out)"
        f" | session: ${tracker.total_cost:.4f}[/dim]"
    )


# ── Slash Commands ────────────────────────────────────────────────────────────

def session_path(config: AgentConfig) -> Path:
    from pathlib import Path
    return Path(config.project_dir) / ".agentcode_session.json"


def handle_slash_command(
    cmd: str, config: AgentConfig, conversation: Conversation, project_config: dict | None = None
) -> bool:
    """Handle /commands. Returns True if the command was handled."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif command == "/clear":
        conversation.messages.clear()
        session_path(config).unlink(missing_ok=True)
        console.print("[success]✓ Conversation cleared[/success]")
        return True

    elif command == "/model":
        if arg:
            config.model = arg
            if config.router:
                config.router.enabled = False
            console.print(f"[success]✓ Switched to model: {arg} (routing disabled)[/success]")
        else:
            console.print(f"[info]Current model: {config.model}[/info]")
            console.print("[dim]Usage: /model <model_name>[/dim]")
            console.print("[dim]Examples: claude-sonnet-4-6, gpt-4o, ollama/llama3[/dim]")
        return True

    elif command == "/route":
        if not config.router:
            console.print("[warning]Router not available.[/warning]")
            return True
        if arg in ("on", "off"):
            config.router.enabled = arg == "on"
            console.print(f"[success]✓ Routing {'enabled' if config.router.enabled else 'disabled'}[/success]")
        else:
            status = "[bold green]enabled[/bold green]" if config.router.enabled else "[dim]disabled[/dim]"
            console.print(f"  Routing: {status}")
            console.print(f"  Provider: [bold]{config.router.provider}[/bold]")
            for tier in config.router.get_tiers():
                console.print(
                    f"  [dim]{tier.tier:6}[/dim]  [bold]{tier.name}[/bold]"
                    f"  [dim](${tier.input_cost_per_mtok:.2f}/${tier.output_cost_per_mtok:.2f} per 1M tok)[/dim]"
                )
            console.print("[dim]  Usage: /route on | /route off[/dim]")
        return True

    elif command == "/cost":
        if not config.router:
            console.print("[warning]Cost tracking requires routing to be enabled.[/warning]")
            return True
        console.print(f"[info]{config.router.cost_tracker.summary()}[/info]")
        return True

    elif command == "/compact":
        conversation.compact(max_tokens=0, model=config.model)
        console.print("[success]✓ Context compacted[/success]")
        return True

    elif command == "/help":
        show_banner(config, project_config)
        return True

    elif command == "/init":
        _create_agentcode_template(config.project_dir)
        return True

    elif command == "/tokens":
        est = conversation.token_estimate()
        console.print(f"[info]Estimated tokens in context: ~{est:,}[/info]")
        return True

    elif command == "/mcp":
        handle_mcp_command(arg.strip(), config)
        return True

    elif command == "/settings":
        _show_settings(config)
        return True

    return False


# ── /settings display ────────────────────────────────────────────────────────

def _show_settings(config: AgentConfig) -> None:
    """Print the resolved settings and which files were loaded."""
    sources = settings_sources(config.project_dir)
    active = [v for v in sources.values() if v]
    if active:
        for path in active:
            console.print(f"  [dim]Loaded:[/dim] {path}")
    else:
        console.print("  [dim]No settings.json found — showing built-in defaults[/dim]")
        console.print(f"  [dim]Run with --init-settings to create one.[/dim]")

    s = config.settings
    if s is None:
        return

    console.print()
    perms = s.permissions
    auto_all = "[bold green]true[/bold green]" if perms.auto_approve_all else "[dim]false[/dim]"
    console.print(f"  [bold]permissions.auto_approve_all[/bold]  {auto_all}")
    console.print(f"  [bold]permissions.auto_approve[/bold]")
    for name in perms.auto_approve:
        console.print(f"    [dim]·[/dim] {name}")
    deny_str = ", ".join(perms.deny) if perms.deny else "[dim](none)[/dim]"
    console.print(f"  [bold]permissions.deny[/bold]              {deny_str}")

    console.print()
    m = s.model
    console.print(f"  [bold]model.default[/bold]    {m.default}")
    routing_str = "[bold green]true[/bold green]" if m.routing else "[dim]false[/dim]"
    console.print(f"  [bold]model.routing[/bold]    {routing_str}")
    for tier, val in [("light", m.light), ("medium", m.medium), ("heavy", m.heavy)]:
        if val:
            console.print(f"  [bold]model.{tier}[/bold]     {val}")

    console.print()
    lim = s.limits
    console.print(f"  [bold]limits.max_file_size[/bold]       {lim.max_file_size:,} bytes")
    console.print(f"  [bold]limits.max_output[/bold]          {lim.max_output:,} chars")
    console.print(f"  [bold]limits.max_search_results[/bold]  {lim.max_search_results}")
    console.print(f"  [bold]limits.max_iterations[/bold]      {lim.max_iterations}")

    if s.hooks:
        console.print()
        console.print("  [bold]hooks:[/bold]")
        for k, v in s.hooks.items():
            console.print(f"    [dim]{k}:[/dim] {v}")


# ── MCP command ───────────────────────────────────────────────────────────────

MCP_PRESETS = {
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_prompts": {"GITHUB_TOKEN": "GitHub personal access token (github.com/settings/tokens)"},
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "env_prompts": {},
    },
    "postgres": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres"],
        "env_prompts": {"POSTGRES_CONNECTION_STRING": "Postgres connection string (postgresql://user:pass@host/db)"},
    },
    "sqlite": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite"],
        "env_prompts": {"SQLITE_DB_PATH": "Path to SQLite database file"},
    },
}


def _load_mcp_config_file(project_dir: str) -> dict:
    from pathlib import Path
    path = Path(project_dir) / ".agentcode" / "mcp.json"
    if path.exists():
        try:
            import json
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"mcpServers": {}}


def _save_mcp_config_file(project_dir: str, data: dict) -> None:
    from pathlib import Path
    import json
    path = Path(project_dir) / ".agentcode" / "mcp.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def handle_mcp_command(arg: str, config: AgentConfig) -> None:
    parts = arg.split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    name = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        if config.mcp_manager and config.mcp_manager.server_names():
            for server in config.mcp_manager.server_names():
                tool_count = len([k for k in config.mcp_manager._tool_map if k.startswith(f"mcp__{server}__")])
                console.print(f"  [bold green]✓[/bold green] [bold]{server}[/bold] ({tool_count} tools)")
        else:
            console.print("[dim]No MCP servers connected. Use /mcp add <server> to connect one.[/dim]")
            console.print(f"[dim]Available presets: {', '.join(MCP_PRESETS.keys())}[/dim]")

    elif sub == "add":
        if not name:
            console.print(f"[warning]Usage: /mcp add <server>[/warning]")
            console.print(f"[dim]Available presets: {', '.join(MCP_PRESETS.keys())}[/dim]")
            return

        if name not in MCP_PRESETS:
            console.print(f"[warning]Unknown preset '{name}'. Available: {', '.join(MCP_PRESETS.keys())}[/warning]")
            return

        preset = MCP_PRESETS[name]
        server_config = {"command": preset["command"], "args": preset["args"].copy()}

        # Prompt for required env vars
        if preset["env_prompts"]:
            server_config["env"] = {}
            for var, prompt in preset["env_prompts"].items():
                try:
                    value = console.input(f"  [bold]{prompt}:[/bold] ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[warning]Cancelled.[/warning]")
                    return
                if not value:
                    console.print("[warning]Cancelled — value required.[/warning]")
                    return
                server_config["env"][var] = value

        # Save to config file
        data = _load_mcp_config_file(config.project_dir)
        data["mcpServers"][name] = server_config
        _save_mcp_config_file(config.project_dir, data)

        # Hot-connect without restart
        from mcp_client import MCPManager
        if config.mcp_manager is None:
            config.mcp_manager = MCPManager()
        try:
            config.mcp_manager.connect(name, server_config)
            tool_count = len([k for k in config.mcp_manager._tool_map if k.startswith(f"mcp__{name}__")])
            console.print(f"[success]✓ Connected to {name} ({tool_count} tools available)[/success]")
        except Exception as e:
            console.print(f"[error]Failed to connect to {name}: {e}[/error]")
            # Roll back config
            data["mcpServers"].pop(name, None)
            _save_mcp_config_file(config.project_dir, data)

    elif sub == "remove":
        if not name:
            console.print("[warning]Usage: /mcp remove <server>[/warning]")
            return

        data = _load_mcp_config_file(config.project_dir)
        if name not in data.get("mcpServers", {}):
            console.print(f"[warning]Server '{name}' not found in config.[/warning]")
            return

        data["mcpServers"].pop(name)
        _save_mcp_config_file(config.project_dir, data)

        # Remove from running manager
        if config.mcp_manager:
            keys_to_remove = [k for k in config.mcp_manager._tool_map if k.startswith(f"mcp__{name}__")]
            for k in keys_to_remove:
                config.mcp_manager._tool_map.pop(k)
            config.mcp_manager._servers.pop(name, None)

        console.print(f"[success]✓ Removed {name}[/success]")

    else:
        console.print("  [bold]/mcp list[/bold]             — show connected servers")
        console.print("  [bold]/mcp add <server>[/bold]     — connect a server")
        console.print("  [bold]/mcp remove <server>[/bold]  — disconnect a server")
        console.print(f"  [dim]Available presets: {', '.join(MCP_PRESETS.keys())}[/dim]")


# ── AGENTCODE.md Template ─────────────────────────────────────────────────────

AGENTCODE_TEMPLATE = """\
# AGENTCODE.md — Project Configuration for AgentCode

## Project Overview
<!-- Brief description of what this project does -->
This is a [describe your project here].

## Tech Stack
<!-- Languages, frameworks, and key dependencies -->
- Language: Python 3.11+
- Framework: [your framework]
- Key dependencies: [list them]

## Coding Standards
<!-- Rules the agent should follow when writing code -->
- Use type hints for all function signatures
- Write docstrings for all public functions
- Follow PEP 8 style conventions
- Keep functions under 50 lines where possible

## File Structure
<!-- Help the agent understand your project layout -->
- `src/` — main source code
- `tests/` — test files (mirror src/ structure)
- `docs/` — documentation

## Testing
<!-- How to run tests and what framework you use -->
- Framework: pytest
- Run tests: `pytest tests/ -v`
- Always write tests for new functions

## Preferences
<!-- Your personal preferences for how the agent works -->
- Prefer descriptive variable names over abbreviations
- Use f-strings for string formatting
- Add comments for complex logic, skip obvious ones
- When in doubt, ask before making breaking changes

## Do NOT
<!-- Things the agent should avoid -->
- Do not modify configuration files without asking
- Do not install new dependencies without approval
- Do not delete files without confirmation
"""


def _create_agentcode_template(project_dir: str):
    """Create an AGENTCODE.md template in the project directory."""
    from pathlib import Path
    target = Path(project_dir) / "AGENTCODE.md"

    if target.exists():
        console.print(f"[warning]⚠ AGENTCODE.md already exists at {target}[/warning]")
        return

    target.write_text(AGENTCODE_TEMPLATE)
    console.print(f"[success]✓ Created AGENTCODE.md at {target}[/success]")
    console.print("[dim]  Edit it to define your project's coding standards and preferences.[/dim]")
    console.print("[dim]  Restart AgentCode to load the config.[/dim]")


# ── REPL ──────────────────────────────────────────────────────────────────────

def repl(config: AgentConfig):
    """Interactive read-eval-print loop."""
    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)

    sess = session_path(config)
    conversation.load(sess)
    if conversation.messages:
        console.print(f"[dim]Resumed session ({len(conversation.messages)} messages). Use /clear to start fresh.[/dim]\n")

    show_banner(config, project_config)

    while True:
        try:
            user_input = console.input("[prompt]❯[/prompt] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_slash_command(user_input, config, conversation, project_config):
                continue

        try:
            run_agent_loop(user_input, conversation, config)
            conversation.save(sess)
            display_cost_line(config)
        except KeyboardInterrupt:
            console.print("\n[warning]⚠ Interrupted[/warning]")
            continue
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")
            continue


# ── One-Shot Mode ─────────────────────────────────────────────────────────────

def one_shot(prompt: str, config: AgentConfig):
    """Execute a single prompt and exit."""
    project_config = load_project_config(config.project_dir)
    system_prompt = build_system_prompt(config.project_dir, project_config["combined"])
    conversation = Conversation(system=system_prompt)
    run_agent_loop(prompt, conversation, config)
    display_cost_line(config)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AgentCode — multi-model agentic coding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="One-shot prompt (omit for interactive REPL)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model to use (overrides settings.model.default)",
    )
    parser.add_argument(
        "--no-route",
        action="store_true",
        help="Disable cost-aware model routing (always use --model)",
    )
    parser.add_argument(
        "--auto-approve", "-y",
        action="store_true",
        help="Skip permission prompts for all tool calls (use with caution!)",
    )
    parser.add_argument(
        "--dir", "-d",
        default=os.getcwd(),
        help="Project directory to operate in (default: current dir)",
    )
    parser.add_argument(
        "--init-settings",
        action="store_true",
        help="Create a starter .agentcode/settings.json and exit",
    )

    args = parser.parse_args()

    project_dir = os.path.abspath(args.dir)
    os.chdir(project_dir)

    # Handle --init-settings before anything else
    if args.init_settings:
        path, created = generate_starter_settings(project_dir)
        if created:
            console.print(f"[success]✓ Created {path}[/success]")
            console.print()
            console.print("[dim]Settings reference:[/dim]")
            for line in SETTINGS_HELP.strip().splitlines():
                console.print(f"  [dim]{line}[/dim]")
        else:
            console.print(f"[warning]⚠  Settings already exist at {path}[/warning]")
            console.print("[dim]  Edit it directly or delete it and re-run --init-settings.[/dim]")
        return

    # Build CLI overrides dict so they take highest priority in merge
    cli_overrides: dict = {}
    if args.auto_approve:
        cli_overrides.setdefault("permissions", {})["auto_approve_all"] = True
    if args.no_route:
        cli_overrides.setdefault("model", {})["routing"] = False

    settings = load_settings(project_dir, cli_overrides or None)

    # Apply configurable limits to tools module
    configure_limits(settings.limits)

    # Resolve model: CLI flag > env var > settings.model.default
    model = (
        args.model
        or os.environ.get("AGENTCODE_MODEL")
        or settings.model.default
    )

    # Resolve routing
    routing_enabled = settings.model.routing  # already patched by cli_overrides if --no-route

    # Build router
    temp_router = ModelRouter(default_model=model)
    provider = temp_router.detect_provider(model)
    router = ModelRouter(
        provider=provider,
        default_model=model,
        enabled=routing_enabled,
    )

    # Resolve max_iterations: env var takes precedence for backward compat
    max_iterations = int(
        os.environ.get("AGENTCODE_MAX_ITERATIONS", settings.limits.max_iterations)
    )

    config = AgentConfig(
        model=model,
        auto_approve=settings.permissions.auto_approve_all,
        project_dir=project_dir,
        max_iterations=max_iterations,
        router=router,
        mcp_manager=create_mcp_manager(project_dir),
        settings=settings,
    )

    if args.prompt:
        one_shot(args.prompt, config)
    else:
        repl(config)


if __name__ == "__main__":
    main()
