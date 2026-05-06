# AgentCode

An open, multi-model agentic coding assistant — available as a CLI and VS Code extension. Inspired by Claude Code.

Works with Claude, GPT, Gemini, and any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers).

---

## Choose your setup

### Option A: CLI only

```bash
pip install agentcode-cli
```

Add your API key to a `.env` file in your project:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # optional
GEMINI_API_KEY=...           # optional
```

Run:

```bash
agentcode                        # interactive REPL
agentcode "fix the failing tests" # one-shot mode
agentcode --model gpt-4o         # use a specific model
```

---

### Option B: VS Code Extension

> The extension requires the CLI installed as a backend. Install it first.

**Step 1 — Install the CLI:**
```bash
pip install agentcode-cli
```

**Step 2 — Install the extension:**
- Download `agentcode-1.0.0.vsix` from the [GitHub releases page](https://github.com/vigp17/AgentCode/releases)
- In VS Code: `Cmd+Shift+X` → `...` → **Install from VSIX...**

**Step 3 — Add your API key:**
`Cmd+,` → search **AgentCode** → paste your key into `agentcode.anthropicApiKey`

**Step 4 — Open the chat panel:**
Press `Cmd+Shift+A`

#### Extension settings

| Setting | Description |
|---------|-------------|
| `agentcode.anthropicApiKey` | Anthropic API key |
| `agentcode.openaiApiKey` | OpenAI API key |
| `agentcode.geminiApiKey` | Google Gemini API key |
| `agentcode.model` | Default model (e.g. `claude-sonnet-4-6`) |
| `agentcode.executablePath` | Path to `agentcode` if not on PATH |

#### Extension commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `AgentCode: Open` | `Cmd+Shift+A` | Open the chat panel |
| `AgentCode: Ask about selection` | Right-click | Ask about highlighted code |
| `AgentCode: Explain this file` | Right-click | Explain the current file |

#### Extension features

- **Live streaming** — responses stream in real time
- **Model picker** — switch between Claude, GPT, and Gemini models from the dropdown in the header
- **Active file context** — your current file is sent automatically, no copy-pasting needed
- **Diff viewer** — edit tool calls open VS Code's native diff viewer so you can review changes before applying
- **Right-click actions** — ask about selected code or explain a file directly from the editor

---

## Supported Models

| Provider | Model | API Key |
|----------|-------|---------|
| Anthropic | `claude-sonnet-4-6` (default) | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-opus-4-7` | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-5.5` | `OPENAI_API_KEY` |
| Google | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |

---

## Cost-Aware Routing

AgentCode automatically picks the cheapest model that can handle the task:

| Tier | Anthropic | OpenAI | Gemini |
|------|-----------|--------|--------|
| Light | Haiku 4.5 | GPT-4o Mini | Gemini 2.0 Flash |
| Medium | Sonnet 4.6 | GPT-4o | Gemini 2.5 Flash |
| Heavy | Opus 4.7 | GPT-5.5 | Gemini 2.5 Pro |

Simple questions go to cheap/fast models. Complex multi-file tasks go to powerful ones. Use `--no-route` to always use the specified model.

---

## Tools

### File & Shell

| Tool | Description | Permission |
|------|-------------|------------|
| `read_file` | Read file contents with line numbers | Auto |
| `write_file` | Create or overwrite a file | Ask |
| `edit_file` | Surgical find-and-replace edit | Ask |
| `run_command` | Execute a bash command | Ask |
| `list_directory` | Tree view of directory structure | Auto |
| `search_files` | Find files by glob pattern | Auto |
| `search_text` | Grep for text across files | Auto |

### Git

| Tool | Description | Permission |
|------|-------------|------------|
| `git_status` | Show working tree status | Auto |
| `git_diff` | Show staged or unstaged changes | Auto |
| `git_log` | Show recent commit history | Auto |
| `git_commit` | Stage files and create a commit | Ask |
| `git_branch` | List, create, or switch branches | Ask |
| `git_push` | Push commits to a remote | Ask |

### Subagents

| Tool | Description | Permission |
|------|-------------|------------|
| `spawn_subagents` | Run multiple agents in parallel on subtasks | Auto |

**Permission model:** Read-only tools auto-approve. Write/execute tools ask before running (unless `--auto-approve` / `-y` is set).

---

## Slash Commands (CLI only)

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model on the fly |
| `/route` | Show or toggle cost-aware routing |
| `/cost` | Show session cost breakdown |
| `/mcp` | Manage MCP server connections |
| `/mcp list` | Show connected servers |
| `/mcp add <server>` | Connect a server |
| `/mcp remove <server>` | Disconnect a server |
| `/clear` | Reset conversation and delete saved session |
| `/compact` | Force LLM-powered context compaction |
| `/tokens` | Show estimated token usage |
| `/init` | Create an AGENTCODE.md template |
| `/settings` | Show resolved settings |
| `/help` | Show help |
| `/exit` | Quit |

---

## Session Persistence

Conversations are automatically saved to `.agentcode_session.json` in your project directory and resumed on next launch. Use `/clear` to start fresh.

---

## MCP Support (CLI only)

Connect to any [MCP server](https://modelcontextprotocol.io) using `/mcp add`:

```
/mcp add github       # prompts for GitHub token
/mcp add filesystem   # no credentials needed
/mcp add postgres     # prompts for connection string
/mcp add sqlite       # prompts for database path
```

Config is saved to `.agentcode/mcp.json` and reloaded on next launch.

**Advanced** — edit `.agentcode/mcp.json` directly for custom servers:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "@myorg/mcp-server"],
      "env": {"API_KEY": "..."}
    }
  }
}
```

Global config goes in `~/.agentcode/mcp.json`.

---

## Hooks

Run shell commands before or after any tool call. Create `.agentcode/hooks.json`:

```json
{
  "post_edit_file": "prettier --write \"$AGENTCODE_PATH\"",
  "post_write_file": "prettier --write \"$AGENTCODE_PATH\"",
  "pre_run_command": "echo \"Running: $AGENTCODE_COMMAND\""
}
```

Supported keys: `pre_<toolname>`, `post_<toolname>`, `pre_tool` / `post_tool` (wildcard). Global hooks go in `~/.agentcode/hooks.json`.

---

## AGENTCODE.md

AgentCode loads project instructions from `AGENTCODE.md` in your project directory and injects them into the system prompt automatically. Run `/init` to generate a starter template.

---

## Subagents

AgentCode can spawn parallel agents for independent subtasks:

> "Analyze agent.py, router.py, and tools.py in parallel and summarize each one"

The agent calls `spawn_subagents` internally, runs up to 5 agents in parallel, and returns combined results.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  cli.py (UI)                    │
│  REPL loop · slash commands · Rich terminal UI  │
│  AGENTCODE.md · session persistence             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               agent.py (Brain)                  │
│  Agentic loop · context management · permissions│
│  Hooks · subagents · LLM-powered compaction     │
│                                                 │
│   while needs_follow_up:                        │
│     1. Send messages + tools → LLM              │
│     2. If tool_calls → execute, append, loop    │
│     3. If text only  → done                     │
│                                                 │
│   LiteLLM ──→ Claude / GPT / Gemini             │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼───────────┐   ┌─────────▼───────────┐
│    tools.py (Hands) │   │  mcp_client.py       │
│  read_file          │   │  Connect to MCP      │
│  write_file         │   │  servers and expose  │
│  edit_file          │   │  their tools to the  │
│  run_command        │   │  agent loop.         │
│  list_directory     │   └─────────────────────┘
│  search_files       │
│  search_text        │
│  git_status/diff    │
│  git_log/commit     │
│  git_branch/push    │
│  spawn_subagents    │
└─────────────────────┘
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTCODE_MODEL` | Default model | `claude-sonnet-4-6` |
| `AGENTCODE_MAX_ITERATIONS` | Max tool-call iterations per turn | `25` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GEMINI_API_KEY` | Google Gemini API key | — |

---

## How to Extend

### Add a new tool

1. Add the function schema to `TOOL_DEFINITIONS` in `tools.py`
2. Implement the function
3. Register it in `TOOL_MAP`
4. If it requires approval, omit it from `permissions.auto_approve` in `.agentcode/settings.json`

---

## Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## License

MIT
