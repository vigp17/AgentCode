"""
AgentCode - Tool definitions and execution.

These are the "hands" of the agent — the actions it can take in the real world.
Each tool maps to a function definition (for the LLM) and an implementation.
"""

import os
import subprocess
import glob as glob_module
from pathlib import Path


# ── Configurable limits (set at startup via configure_limits) ─────────────────

_LIMITS: dict = {
    "max_file_size": 1_000_000,
    "max_output": 50_000,
    "max_search_results": 100,
}


def configure_limits(limits) -> None:
    """Apply a LimitsSettings object to tool execution. Call once at startup."""
    _LIMITS["max_file_size"] = limits.max_file_size
    _LIMITS["max_output"] = limits.max_output
    _LIMITS["max_search_results"] = limits.max_search_results


# ── Tool Definitions (OpenAI-compatible function calling schema) ──────────────
# LiteLLM translates these to the right format for each provider.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Use this to understand existing code "
                "before making changes. Returns the file content with line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional: first line to read (1-indexed). Omit to read from start.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional: last line to read (1-indexed). Omit to read to end.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file or overwrite an existing file with the given content. "
                "Use this for creating new files. For editing existing files, prefer "
                "edit_file to make surgical changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Make a targeted edit to an existing file by replacing a unique string "
                "with new content. The old_string must match exactly and appear only once "
                "in the file. Always read the file first to get the exact text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace. Must be unique in the file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace old_string with. Use empty string to delete.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Execute a bash command in the user's shell. Use for running tests, "
                "installing packages, git operations, or any shell task. Commands run "
                "in the project directory. Stderr and stdout are both captured."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds to wait (default: 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List files and directories at a given path. Shows file sizes and types. "
                "Use to understand project structure before diving into files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many levels deep to list (default: 2).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for files by name pattern (glob). Use to find files matching "
                "a pattern like '*.py' or 'test_*.js'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g., '**/*.py', 'src/**/*.ts').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search from. Defaults to '.'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Search for a text pattern (regex) across files in a directory. "
                "Similar to 'grep -rn'. Returns matching lines with file paths and "
                "line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to '.'.",
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob to include (e.g., '*.py'). Optional.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the git working tree status — staged, unstaged, and untracked files.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "Show changes in the working tree. Use staged=true to see changes "
                "that are staged (indexed) for the next commit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "staged": {
                        "type": "boolean",
                        "description": "If true, show staged changes. Default false shows unstaged changes.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent commit history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of commits to show (default: 10).",
                    },
                    "oneline": {
                        "type": "boolean",
                        "description": "If true, show one line per commit.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": (
                "Stage files and create a commit. By default stages all modified tracked files. "
                "Use files to stage specific paths, or add_all=true to include untracked files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The commit message.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific file paths to stage. Omit to stage all modified tracked files.",
                    },
                    "add_all": {
                        "type": "boolean",
                        "description": "If true, stage everything including untracked files (git add -A).",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List, create, or switch git branches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "switch"],
                        "description": "list all branches, create a new branch, or switch to an existing one.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Branch name — required for create and switch.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Push commits to a remote repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote": {
                        "type": "string",
                        "description": "Remote name (default: origin).",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch to push. Omit to push the current branch.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagents",
            "description": (
                "Spawn multiple independent agents to work on subtasks in parallel. "
                "Use when a task can be split into independent parts that don't depend on each other — "
                "e.g. analyze multiple files simultaneously, run parallel searches, or investigate "
                "different parts of the codebase at once. Each subagent gets its own context and "
                "runs the full agentic loop. Results are returned combined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of independent subtask prompts to run in parallel. Keep each task self-contained.",
                    },
                },
                "required": ["tasks"],
            },
        },
    },
]


# ── Tool Implementations ─────────────────────────────────────────────────────

def _read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file and return its contents with line numbers."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        if p.stat().st_size > _LIMITS["max_file_size"]:
            return f"Error: File too large ({p.stat().st_size} bytes). Use start_line/end_line."

        lines = p.read_text(errors="replace").splitlines()
        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(lines)
        selected = lines[start:end]

        numbered = [f"{i + start + 1:>5} │ {line}" for i, line in enumerate(selected)]
        header = f"── {path} ({len(lines)} lines total, showing {start+1}-{end}) ──"
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"✓ Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace a unique string in a file."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text(errors="replace")
        count = content.count(old_string)

        if count == 0:
            return f"Error: old_string not found in {path}. Read the file first to get exact text."
        if count > 1:
            return f"Error: old_string appears {count} times in {path}. Make it more specific."

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content)
        return f"✓ Edited {path} (replaced 1 occurrence)"
    except Exception as e:
        return f"Error editing file: {e}"


def _run_command(command: str, timeout: int = 30) -> str:
    """Execute a bash command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"EXIT CODE: {result.returncode}")

        output = "\n".join(output_parts)
        if len(output) > _LIMITS["max_output"]:
            half = _LIMITS["max_output"] // 2
            output = output[:half] + "\n\n... [truncated] ...\n\n" + output[-half:]
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"


def _list_directory(path: str = ".", depth: int = 2) -> str:
    """List directory contents up to a given depth."""
    try:
        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"Error: Directory not found: {path}"
        if not base.is_dir():
            return f"Error: Not a directory: {path}"

        lines = [f"📁 {base}/"]
        _tree(base, lines, prefix="", depth=depth, max_depth=depth)
        return "\n".join(lines[:500])  # Cap output
    except Exception as e:
        return f"Error listing directory: {e}"


def _tree(directory: Path, lines: list, prefix: str, depth: int, max_depth: int):
    """Recursively build a tree listing."""
    if depth <= 0:
        return

    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}
    try:
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        lines.append(f"{prefix}[permission denied]")
        return

    entries = [e for e in entries if e.name not in skip and not e.name.startswith(".")]
    for i, entry in enumerate(entries[:50]):  # Cap per-level
        connector = "└── " if i == len(entries) - 1 else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}📁 {entry.name}/")
            extension = "    " if i == len(entries) - 1 else "│   "
            _tree(entry, lines, prefix + extension, depth - 1, max_depth)
        else:
            size = entry.stat().st_size
            size_str = _human_size(size)
            lines.append(f"{prefix}{connector}{entry.name} ({size_str})")


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching a glob pattern."""
    try:
        base = Path(path).expanduser().resolve()
        matches = sorted(glob_module.glob(str(base / pattern), recursive=True))
        cap = _LIMITS["max_search_results"]
        matches = [str(Path(m).relative_to(base)) for m in matches[:cap]]
        if not matches:
            return f"No files matching '{pattern}' in {path}"
        return f"Found {len(matches)} files:\n" + "\n".join(matches)
    except Exception as e:
        return f"Error searching files: {e}"


def _search_text(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Search for text pattern across files using grep."""
    try:
        cmd = ["grep", "-rn", "--color=never", "-E", pattern]
        if include:
            cmd.extend(["--include", include])
        cmd.append(path)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, cwd=os.getcwd()
        )
        output = result.stdout
        if not output:
            return f"No matches for pattern '{pattern}' in {path}"
        lines = output.strip().split("\n")
        cap = _LIMITS["max_search_results"]
        if len(lines) > cap:
            return "\n".join(lines[:cap]) + f"\n... and {len(lines) - cap} more matches"
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching text: {e}"


# ── Git Tool Implementations ─────────────────────────────────────────────────

def _git_run(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd())


def _git_status() -> str:
    try:
        r = _git_run(["git", "status"])
        if r.returncode != 0:
            return f"Error: {r.stderr.strip() or 'git status failed'}"
        return r.stdout or "Nothing to report."
    except FileNotFoundError:
        return "Error: git not found."
    except Exception as e:
        return f"Error: {e}"


def _git_diff(staged: bool = False) -> str:
    try:
        cmd = ["git", "diff"] + (["--staged"] if staged else [])
        r = _git_run(cmd)
        if r.returncode != 0:
            return f"Error: {r.stderr.strip() or 'git diff failed'}"
        if not r.stdout:
            return "No staged changes." if staged else "No unstaged changes."
        output = r.stdout
        if len(output) > _LIMITS["max_output"]:
            output = output[:_LIMITS["max_output"]] + "\n... [truncated]"
        return output
    except Exception as e:
        return f"Error: {e}"


def _git_log(n: int = 10, oneline: bool = False) -> str:
    try:
        fmt = "--oneline" if oneline else "--pretty=format:%h %s (%an, %ar)"
        r = _git_run(["git", "log", f"-{n}", fmt])
        if r.returncode != 0:
            return f"Error: {r.stderr.strip() or 'git log failed'}"
        return r.stdout.strip() or "No commits yet."
    except Exception as e:
        return f"Error: {e}"


def _git_commit(message: str, files: list[str] | None = None, add_all: bool = False) -> str:
    try:
        if add_all:
            stage = _git_run(["git", "add", "-A"])
        elif files:
            stage = _git_run(["git", "add", *files])
        else:
            stage = _git_run(["git", "add", "-u"])

        if stage.returncode != 0:
            return f"Error staging files: {stage.stderr.strip()}"

        r = _git_run(["git", "commit", "-m", message])
        if r.returncode != 0:
            return f"Error committing: {r.stderr.strip() or r.stdout.strip()}"
        return r.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def _git_branch(action: str = "list", name: str | None = None) -> str:
    try:
        if action == "list":
            r = _git_run(["git", "branch", "-a"])
        elif action == "create":
            if not name:
                return "Error: branch name required for create."
            r = _git_run(["git", "checkout", "-b", name])
        elif action == "switch":
            if not name:
                return "Error: branch name required for switch."
            r = _git_run(["git", "checkout", name])
        else:
            return f"Error: unknown action '{action}'. Use list, create, or switch."

        if r.returncode != 0:
            return f"Error: {r.stderr.strip() or 'git branch operation failed'}"
        return r.stdout.strip() or r.stderr.strip() or "Done."
    except Exception as e:
        return f"Error: {e}"


def _git_push(remote: str = "origin", branch: str | None = None) -> str:
    try:
        cmd = ["git", "push", remote] + ([branch] if branch else [])
        r = _git_run(cmd, timeout=30)
        if r.returncode != 0:
            return f"Error: {r.stderr.strip() or 'git push failed'}"
        return r.stdout.strip() or r.stderr.strip() or "Pushed successfully."
    except Exception as e:
        return f"Error: {e}"


# ── Tool Router ───────────────────────────────────────────────────────────────

TOOL_MAP = {
    "read_file": _read_file,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "run_command": _run_command,
    "list_directory": _list_directory,
    "search_files": _search_files,
    "search_text": _search_text,
    "git_status": _git_status,
    "git_diff": _git_diff,
    "git_log": _git_log,
    "git_commit": _git_commit,
    "git_branch": _git_branch,
    "git_push": _git_push,
}


def execute_tool(name: str, args: dict) -> str:
    """Route a tool call to its implementation."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {e}"
