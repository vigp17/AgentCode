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
agentcode                          # interactive REPL
agentcode "fix the failing tests"  # one-shot mode
agentcode --model gpt-5.6-terra    # use a specific model
```

All flags:

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Model to use (overrides settings and `AGENTCODE_MODEL`) |
| `--no-route` | Disable cost-aware routing — always use the specified model |
| `--auto-approve`, `-y` | Skip permission prompts (use with caution) |
| `--dir`, `-d` | Project directory to operate in (default: current dir) |
| `--init-settings` | Create a starter `.agentcode/settings.json` and exit |
| `--server` | JSON stdio server mode (used by the VS Code extension) |

---

### Option B: VS Code Extension

> The extension requires the CLI installed as a backend. Install it first.

**Step 1 — Install the CLI:**
```bash
pip install agentcode-cli
```

**Step 2 — Install the extension:**
- Search **AgentCode** in the VS Code Marketplace and install, or
- Download the latest `.vsix` from the [GitHub releases page](https://github.com/vigp17/AgentCode/releases) and install via `Cmd+Shift+X` → `...` → **Install from VSIX...**

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
| `agentcode.model` | Default model (e.g. `claude-sonnet-5`) |
| `agentcode.executablePath` | Path to `agentcode` if not on PATH |
| `agentcode.inlineCompletions.enabled` | Enable/disable inline completions (default: `true`) |

#### Extension commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `AgentCode: Open` | `Cmd+Shift+A` | Open the chat panel |
| `AgentCode: Ask about selection` | Right-click | Ask about highlighted code |
| `AgentCode: Explain this file` | Right-click | Explain the current file |
| `AgentCode: Toggle inline completions` | Command Palette | Enable or disable inline completions |

#### Extension features

- **Inline completions** — AI-powered ghost-text suggestions as you type, powered by Claude Haiku. Press `Tab` to accept
- **Live streaming** — responses stream in real time
- **Model picker** — every Claude, GPT and Gemini model your keys can use, discovered automatically, plus AgentCode's fine-tunes. Filter box and ↻ refresh
- **Active file context** — your current file is sent automatically, no copy-pasting needed
- **Diff viewer** — edit tool calls open VS Code's native diff viewer so you can review changes before applying
- **Right-click actions** — ask about selected code or explain a file directly from the editor

---

## Supported Models

| Provider | Model | API Key |
|----------|-------|---------|
| Anthropic | `claude-sonnet-5` (default) | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-opus-5` | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-fable-5` (opt-in, hardest long-horizon work) | `ANTHROPIC_API_KEY` |
| Anthropic | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` | `OPENAI_API_KEY` |
| Google | `gemini/gemini-3.1-pro-preview`, `gemini/gemini-3.6-flash`, `gemini/gemini-3.5-flash-lite` | `GEMINI_API_KEY` |

Any of LiteLLM's ~3,000 models works. The table above is only the built-in
fallback. Run `/models` in the REPL to see every model your keys can use, with
current prices, then `/model <name>` to switch.

### New models appear automatically

You don't need to add new Anthropic, OpenAI or Google models yourself.
On startup AgentCode asks each provider which models your API key can use:

| Provider | Endpoint | Key |
|----------|----------|-----|
| Anthropic | `GET /v1/models` | `ANTHROPIC_API_KEY` |
| OpenAI | `GET /v1/models` (chat models only, includes your `ft:` fine-tunes) | `OPENAI_API_KEY` |
| Google | `GET /v1beta/models` (models that support `generateContent`) | `GEMINI_API_KEY` |

- **Picker and `/models`.** A model shows up the day the provider ships it,
  with no AgentCode release and no `pip install -U`. In VS Code, open the model
  dropdown and use the filter box, or click **↻ Check for new models**. In the
  CLI, run `/models` or `/models refresh`.
- **Routing.** With `model.auto_tiers` on (the default), each tier uses the
  newest model of its class: newest Haiku, Sonnet and Opus; newest
  Flash-Lite, Flash and Pro; newest Luna, Terra and Sol. A new model AgentCode
  can't price yet is skipped, so cost tracking never silently drops to $0.
  Tiers you set yourself always win.
- **Caching.** Results are cached in `~/.agentcode/models_cache.json` for 24h
  (`AGENTCODE_MODELS_TTL_HOURS`). The cache stores a hash of the key, never
  the key itself. With no key, or offline, AgentCode falls back to LiteLLM's
  model registry. Turn discovery off with `model.auto_discover: false` or
  `AGENTCODE_DISABLE_DISCOVERY=1`.

Prices come from LiteLLM's model registry. To pin a tier to a specific model,
set it in `.agentcode/settings.json`:

```json
{
  "model": {
    "light":  "gpt-5.6-luna",
    "medium": "gpt-5.6-terra",
    "heavy":  "gpt-5.6-sol"
  }
}
```

If you set a model AgentCode can't price (a typo, or one newer than your
LiteLLM registry), it warns at startup and reports costs as `$0.00` rather
than guessing.

### AgentCode's own fine-tuned models

AgentCode has three fine-tuned models, one per routing tier. They are LoRA
fine-tunes of Qwen2.5-Coder trained on AgentCode tool-calling trajectories,
and you can self-host them for $0 per token:

| Model | Base | Tier |
|-------|------|------|
| `agentcode-light`  | Qwen2.5-Coder-1.5B-Instruct | light |
| `agentcode-medium` | Qwen2.5-Coder-7B-Instruct   | medium |
| `agentcode-heavy`  | Qwen2.5-Coder-32B-Instruct  | heavy |

[`finetune/`](finetune/README.md) has the full pipeline: collect trajectories,
prepare them, train with Unsloth (or OpenAI hosted fine-tuning), serve with
vLLM and evaluate. Once a model is served, point AgentCode at it:

```bash
AGENTCODE_FT_MEDIUM_API_BASE=http://localhost:8002/v1 agentcode --model agentcode-medium
```

---

## Cost-Aware Routing

AgentCode automatically picks the cheapest model that can handle the task:

| Tier | Anthropic | OpenAI | Gemini |
|------|-----------|--------|--------|
| Light | newest Haiku | newest Luna | newest Flash-Lite |
| Medium | newest Sonnet | newest Terra | newest Flash |
| Heavy | newest Opus | newest Sol | newest Pro |

The newest model in each class is discovered automatically (see above). If
discovery is off, the fallback is Haiku 4.5 / Sonnet 5 / Opus 5,
GPT-5.6 Luna / Terra / Sol, and Gemini 3.5 Flash-Lite / 3.6 Flash / 3.1 Pro.

Simple questions go to cheap/fast models. Complex multi-file tasks go to powerful ones. Use `--no-route` to always use the specified model.

For the hardest long-horizon agent work, pick **`claude-fable-5`** manually (CLI `--model` or the extension picker). It is not used by auto-routing so routine heavy tasks stay on Opus.

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

**Permission model:** Read-only tools auto-approve. Write/execute tools and
MCP tools ask before running (unless `--auto-approve` / `-y` is set). Tools on
the `permissions.deny` list are always blocked, even with auto-approve on.
Which tools fall into which bucket is configurable — see Settings below.

---

## Settings

Create a settings file with:

```bash
agentcode --init-settings
```

This writes `.agentcode/settings.json` in your project. A global file at
`~/.agentcode/settings.json` is also read; precedence is
**CLI flags → project → global → built-in defaults**. View the merged result
anytime with `/settings`.

```json
{
  "permissions": {
    "auto_approve_all": false,
    "auto_approve": ["read_file", "list_directory", "search_files",
                     "search_text", "git_status", "git_log", "git_diff",
                     "spawn_subagents"],
    "deny": []
  },
  "model": {
    "default": "claude-sonnet-5",
    "routing": true,
    "light": null, "medium": null, "heavy": null,
    "auto_discover": true,
    "auto_tiers": true
  },
  "limits": {
    "max_file_size": 1000000,
    "max_output": 50000,
    "max_search_results": 100,
    "max_iterations": 25
  },
  "hooks": {},
  "finetuned": {},
  "training": { "log_trajectories": false }
}
```

| Key | Meaning |
|-----|---------|
| `permissions.auto_approve_all` | `true` skips all permission prompts (like `-y`) |
| `permissions.auto_approve` | Tools that run without asking (default: read-only tools) |
| `permissions.deny` | Tools that are always blocked — beats every other setting |
| `model.default` | Default model string |
| `model.routing` | Enable cost-aware routing |
| `model.light/medium/heavy` | Override the model for a routing tier |
| `model.auto_discover` | List every model your API keys can use (default `true`) |
| `model.auto_tiers` | Route each tier to the newest discovered model of its class (default `true`) |
| `finetuned` | Endpoints for `agentcode-light/medium/heavy` or your own fine-tunes ([details](finetune/README.md#5-connect-agentcode)) |
| `training.log_trajectories` | Save finished turns as fine-tuning data (default `false`) |
| `limits.*` | File-size, output, search, and iteration caps |
| `hooks` | Same format as `hooks.json` (see Hooks) |

To gate an MCP tool or approve it permanently, use its full name, e.g.
`"deny": ["mcp__github__delete_repository"]` or
`"auto_approve": ["mcp__filesystem__read_file"]`.

---

## Slash Commands (CLI only)

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model on the fly |
| `/models [filter]` | Models your keys can use (auto-discovered) and live prices |
| `/models refresh` | Re-check providers for newly released models |
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

**Security notes:**
- Credentials in `mcp.json` are stored in plaintext. AgentCode sets the file
  to mode `600` and adds `.agentcode/` to your `.gitignore` automatically, but
  prefer scoped, revocable tokens where the service offers them.
- MCP tools ask for permission before running, like built-in write tools. Add
  specific tools to `permissions.auto_approve` or `permissions.deny` in
  settings.json to change that (see Settings).

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

Subagents inherit the session's permission settings, and their approval
prompts are serialized so they never talk over each other. By default they
cannot spawn subagents of their own — raise `AGENTCODE_MAX_SUBAGENT_DEPTH` if
you want deeper nesting.

---

## Architecture

```
┌────────────────────────┐      ┌────────────────────────┐
│      cli.py (UI)       │      │  server.py (VS Code)   │
│  REPL · slash commands │      │  JSON stdio protocol   │
│  Rich terminal UI      │      │  for the extension     │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
┌───────────▼───────────────────────────────▼────────────┐
│                    agent.py (Brain)                    │
│  Agentic loop · context compaction · permissions       │
│  Hooks · subagents (depth-capped) · settings.py config │
│                                                        │
│   while needs_follow_up:                               │
│     1. router.py picks the model (cost-aware tiers)    │
│     2. Send messages + tools → LLM (via LiteLLM)       │
│     3. If tool_calls → execute, append, loop           │
│        (xml_tool_parser.py converts inline XML/JSON    │
│         tool calls from open-weight fine-tunes)        │
│     4. If text only  → done                            │
└───────────┬───────────────────────────────┬────────────┘
            │                               │
┌───────────▼─────────┐         ┌───────────▼────────────┐
│  tools.py (Hands)   │         │      mcp_client.py     │
│  read/write/edit    │         │  Connect to MCP        │
│  run_command        │         │  servers and expose    │
│  list/search        │         │  their tools to the    │
│  git operations     │         │  agent loop            │
│  spawn_subagents    │         └────────────────────────┘
└─────────────────────┘
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTCODE_MODEL` | Default model | `claude-sonnet-5` |
| `AGENTCODE_MAX_ITERATIONS` | Max tool-call iterations per turn | `25` |
| `AGENTCODE_MAX_SUBAGENT_DEPTH` | How deep subagents may nest | `1` |
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
