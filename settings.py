"""AgentCode settings — load and merge configuration from global, project, and CLI."""

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_AUTO_APPROVE = [
    "read_file", "list_directory", "search_files", "search_text",
    "git_status", "git_log", "git_diff", "spawn_subagents",
]


@dataclass
class PermissionsSettings:
    auto_approve_all: bool = False
    auto_approve: list[str] = field(default_factory=lambda: list(DEFAULT_AUTO_APPROVE))
    deny: list[str] = field(default_factory=list)


@dataclass
class ModelSettings:
    default: str = "claude-sonnet-5"
    routing: bool = True
    light: str | None = None
    medium: str | None = None
    heavy: str | None = None


@dataclass
class LimitsSettings:
    max_file_size: int = 1_000_000
    max_output: int = 50_000
    max_search_results: int = 100
    max_iterations: int = 25


@dataclass
class Settings:
    permissions: PermissionsSettings = field(default_factory=PermissionsSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    limits: LimitsSettings = field(default_factory=LimitsSettings)
    hooks: dict[str, str] = field(default_factory=dict)


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _dict_to_settings(d: dict) -> "Settings":
    """Construct a Settings from a merged raw dict, filling in defaults for missing keys."""
    perms_d = d.get("permissions", {})
    model_d = d.get("model", {})
    limits_d = d.get("limits", {})
    hooks_d = d.get("hooks", {})

    return Settings(
        permissions=PermissionsSettings(
            auto_approve_all=bool(perms_d.get("auto_approve_all", False)),
            auto_approve=list(perms_d.get("auto_approve", DEFAULT_AUTO_APPROVE)),
            deny=list(perms_d.get("deny", [])),
        ),
        model=ModelSettings(
            default=str(model_d.get("default", "claude-sonnet-5")),
            routing=bool(model_d.get("routing", True)),
            light=model_d.get("light") or None,
            medium=model_d.get("medium") or None,
            heavy=model_d.get("heavy") or None,
        ),
        limits=LimitsSettings(
            max_file_size=int(limits_d.get("max_file_size", 1_000_000)),
            max_output=int(limits_d.get("max_output", 50_000)),
            max_search_results=int(limits_d.get("max_search_results", 100)),
            max_iterations=int(limits_d.get("max_iterations", 25)),
        ),
        hooks=dict(hooks_d),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_settings(project_dir: str, cli_overrides: dict | None = None) -> "Settings":
    """
    Load settings by merging (later layers override earlier ones):
      1. Built-in defaults
      2. ~/.agentcode/settings.json       (global)
      3. <project>/.agentcode/settings.json  (project)
      4. cli_overrides                    (CLI flags, highest priority)
    """
    merged: dict = {}

    global_path = Path.home() / ".agentcode" / "settings.json"
    if global_path.exists():
        merged = _deep_merge(merged, _load_json(global_path))

    project_path = Path(project_dir) / ".agentcode" / "settings.json"
    if project_path.exists():
        merged = _deep_merge(merged, _load_json(project_path))

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return _dict_to_settings(merged)


def settings_sources(project_dir: str) -> dict[str, str | None]:
    """Return paths to settings files that currently exist on disk."""
    global_path = Path.home() / ".agentcode" / "settings.json"
    project_path = Path(project_dir) / ".agentcode" / "settings.json"
    return {
        "global": str(global_path) if global_path.exists() else None,
        "project": str(project_path) if project_path.exists() else None,
    }


# ── Starter template ──────────────────────────────────────────────────────────

STARTER_SETTINGS: dict = {
    "permissions": {
        "auto_approve_all": False,
        "auto_approve": [
            "read_file", "list_directory", "search_files", "search_text",
            "git_status", "git_log", "git_diff", "spawn_subagents"
        ],
        "deny": []
    },
    "model": {
        "default": "claude-sonnet-5",
        "routing": True,
        "light": None,
        "medium": None,
        "heavy": None
    },
    "limits": {
        "max_file_size": 1000000,
        "max_output": 50000,
        "max_search_results": 100,
        "max_iterations": 25
    },
    "hooks": {}
}

SETTINGS_HELP = """\
permissions.auto_approve_all  true to skip all permission prompts (like --auto-approve)
permissions.auto_approve      tools that run without asking (default: read-only tools)
permissions.deny              tools that are always blocked, regardless of other settings
model.default                 default LLM model string (e.g. "claude-sonnet-5")
model.routing                 enable cost-aware model routing (true/false)
model.light/medium/heavy      override the model used for each routing tier
limits.max_file_size          max bytes for read_file (default: 1,000,000)
limits.max_output             max chars for tool output truncation (default: 50,000)
limits.max_search_results     cap on results from search tools (default: 100)
limits.max_iterations         max tool-call loops per turn (default: 25)
hooks                         shell commands run before/after tool calls
                              keys: pre_<tool>, post_<tool>, pre_tool, post_tool\
"""


def generate_starter_settings(project_dir: str) -> tuple[Path, bool]:
    """
    Write a starter settings.json to <project>/.agentcode/settings.json.
    Returns (path, created): created=False if the file already existed.
    """
    path = Path(project_dir) / ".agentcode" / "settings.json"
    existed = path.exists()
    if not existed:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(STARTER_SETTINGS, indent=2))
    return path, not existed
