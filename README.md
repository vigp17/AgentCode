# AgentCode

An open, multi-model agentic coding CLI — inspired by Claude Code.

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

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or
pip install litellm rich python-dotenv mcp

# 2. Add your API keys to .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# 3. Run
python cli.py                           # Interactive REPL
python cli.py "fix the failing tests"   # One-shot mode
python cli.py --model gpt-4o            # Use a different model
```

## Supported Models

Via [LiteLLM](https://docs.litellm.ai/docs/providers), AgentCode supports 100+ models:

| Provider   | Model String                    | API Key Env Var         |
|------------|---------------------------------|-------------------------|
| Anthropic  | `claude-sonnet-4-6` (default)   | `ANTHROPIC_API_KEY`     |
| OpenAI     | `gpt-4o`                        | `OPENAI_API_KEY`        |
| Google     | `gemini/gemini-2.5-pro`         | `GEMINI_API_KEY`        |
| Groq       | `groq/llama-3.3-70b-versatile`  | `GROQ_API_KEY`          |

## Cost-Aware Routing

AgentCode automatically picks the cheapest model that can handle the task:

| Tier   | Anthropic          | OpenAI       | Gemini                  |
|--------|--------------------|--------------|-------------------------|
| Light  | Haiku 4.5          | GPT-4o Mini  | Gemini 2.0 Flash        |
| Medium | Sonnet 4.6         | GPT-4o       | Gemini 2.5 Flash        |
| Heavy  | Opus 4.6           | GPT-5.5      | Gemini 2.5 Pro          |

Simple questions go to cheap/fast models. Complex multi-file tasks go to powerful ones. Use `--no-route` to always use the specified model.

## Tools

### File & Shell

| Tool             | Description                          | Permission |
|------------------|--------------------------------------|------------|
| `read_file`      | Read file contents with line numbers | Auto       |
| `write_file`     | Create or overwrite a file           | Ask        |
| `edit_file`      | Surgical find-and-replace edit       | Ask        |
| `run_command`    | Execute a bash command               | Ask        |
| `list_directory` | Tree view of directory structure     | Auto       |
| `search_files`   | Find files by glob pattern           | Auto       |
| `search_text`    | Grep for text across files           | Auto       |

### Git

| Tool          | Description                              | Permission |
|---------------|------------------------------------------|------------|
| `git_status`  | Show working tree status                 | Auto       |
| `git_diff`    | Show staged or unstaged changes          | Auto       |
| `git_log`     | Show recent commit history               | Auto       |
| `git_commit`  | Stage files and create a commit          | Ask        |
| `git_branch`  | List, create, or switch branches         | Ask        |
| `git_push`    | Push commits to a remote                 | Ask        |

### Subagents

| Tool              | Description                                      | Permission |
|-------------------|--------------------------------------------------|------------|
| `spawn_subagents` | Run multiple agents in parallel on subtasks      | Auto       |

**Permission model:** Read-only tools auto-approve. Write/execute tools ask before running (unless `--auto-approve` / `-y` flag is set).

## Slash Commands

| Command           | Description                                      |
|-------------------|--------------------------------------------------|
| `/model <name>`   | Switch LLM model on the fly (disables routing)   |
| `/route`          | Show or toggle cost-aware routing                |
| `/cost`           | Show session cost breakdown                      |
| `/clear`          | Reset conversation and delete saved session      |
| `/compact`        | Force LLM-powered context compaction             |
| `/tokens`         | Show estimated token usage                       |
| `/init`           | Create an AGENTCODE.md template                  |
| `/help`           | Show help                                        |
| `/exit`           | Quit                                             |

## Session Persistence

Conversations are automatically saved to `.agentcode_session.json` in your project directory and resumed on next launch. Use `/clear` to start fresh.

## Hooks

Run shell commands before or after any tool call. Create `.agentcode/hooks.json`:

```json
{
  "post_edit_file": "prettier --write \"$AGENTCODE_PATH\"",
  "post_write_file": "prettier --write \"$AGENTCODE_PATH\"",
  "pre_run_command": "echo \"Running: $AGENTCODE_COMMAND\""
}
```

Supported keys: `pre_<toolname>`, `post_<toolname>`, `pre_tool` / `post_tool` (wildcard). Tool args are passed as `AGENTCODE_<ARG>=value` env vars. Global hooks go in `~/.agentcode/hooks.json`.

## MCP Support

Connect to any [MCP server](https://modelcontextprotocol.io) by creating `.agentcode/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "ghp_..."}
    }
  }
}
```

MCP tools are exposed as `mcp__<server>__<toolname>` and shown in the banner on startup. Global config goes in `~/.agentcode/mcp.json`.

## Subagents

AgentCode can spawn parallel agents to work on independent subtasks simultaneously. Just ask naturally:

> "Analyze agent.py, router.py, and tools.py in parallel and summarize each one"

The agent calls `spawn_subagents` internally, runs up to 5 agents in parallel, and returns combined results.

## AGENTCODE.md

AgentCode loads project-level instructions from `AGENTCODE.md` in your project directory (and global config from `~/.agentcode/AGENTCODE.md`), injecting them into the system prompt automatically.

Run `/init` to generate a starter template, then edit it to define your project's coding standards, preferences, and constraints.

## How to Extend

### Add a new tool

1. Add the function schema to `TOOL_DEFINITIONS` in `tools.py`
2. Implement the function (e.g., `_my_tool(...)`)
3. Register it in `TOOL_MAP`
4. If it's a write/execute tool, add its name to `WRITE_TOOLS` in `agent.py`

## Environment Variables

| Variable                   | Description                       | Default             |
|----------------------------|-----------------------------------|---------------------|
| `AGENTCODE_MODEL`          | Default model                     | `claude-sonnet-4-6` |
| `AGENTCODE_MAX_ITERATIONS` | Max tool-call iterations per turn | `25`                |
| `ANTHROPIC_API_KEY`        | Anthropic API key                 | —                   |
| `OPENAI_API_KEY`           | OpenAI API key                    | —                   |
| `GEMINI_API_KEY`           | Google Gemini API key             | —                   |

## License

MIT
