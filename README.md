# AgentCode

An open, multi-model agentic coding CLI — inspired by Claude Code.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  cli.py (UI)                    │
│  REPL loop · slash commands · Rich terminal UI  │
│  AGENTCODE.md banner · /init template           │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               agent.py (Brain)                  │
│  Agentic loop · context management · permissions│
│                                                 │
│   while needs_follow_up:                        │
│     1. Send messages + tools → LLM              │
│     2. If tool_calls → execute, append, loop    │
│     3. If text only  → done                     │
│                                                 │
│   LiteLLM ──→ Claude / GPT / Gemini / Ollama    │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               tools.py (Hands)                  │
│  read_file · write_file · edit_file             │
│  run_command · list_directory                   │
│  search_files · search_text                     │
│  git_status · git_diff · git_log                │
│  git_commit · git_branch · git_push             │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install litellm rich

# 2. Set your API key (pick one)
export ANTHROPIC_API_KEY="sk-ant-..."   # For Claude
export OPENAI_API_KEY="sk-..."          # For GPT
# Or use Ollama locally (no key needed)

# 3. Run
python cli.py                           # Interactive REPL
python cli.py "fix the failing tests"   # One-shot mode
python cli.py --model gpt-4o            # Use a different model
python cli.py --model ollama/llama3     # Use a local model
```

## Supported Models

Via [LiteLLM](https://docs.litellm.ai/docs/providers), AgentCode supports 100+ models:

| Provider   | Model String                    | API Key Env Var         |
|------------|---------------------------------|-------------------------|
| Anthropic  | `claude-sonnet-4-6`             | `ANTHROPIC_API_KEY`     |
| OpenAI     | `gpt-4o`                        | `OPENAI_API_KEY`        |
| Google     | `gemini/gemini-2.5-pro`         | `GEMINI_API_KEY`        |
| Ollama     | `ollama/llama3`                 | (none, runs locally)    |
| Groq       | `groq/llama-3.3-70b-versatile`  | `GROQ_API_KEY`          |
| Together   | `together_ai/meta-llama/...`    | `TOGETHER_API_KEY`      |

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

**Permission model:** Read-only tools auto-approve. Write/execute tools ask before running (unless `--auto-approve` / `-y` flag is set).

The agent understands natural language git requests:
- *"show me what changed"* → `git_diff` / `git_status`
- *"commit these changes with a good message"* → inspects diff, then `git_commit`
- *"create a branch called feature/streaming"* → `git_branch`
- *"push to origin"* → `git_push`

## Slash Commands

| Command           | Description                         |
|-------------------|-------------------------------------|
| `/model <name>`   | Switch LLM model on the fly         |
| `/clear`          | Reset conversation history          |
| `/compact`        | Force context window compaction     |
| `/tokens`         | Show estimated token usage          |
| `/init`           | Create an AGENTCODE.md template     |
| `/help`           | Show help                           |
| `/exit`           | Quit                                |

## AGENTCODE.md

AgentCode loads project-level instructions from `AGENTCODE.md` in your project directory (and global config from `~/.agentcode/AGENTCODE.md`), injecting them into the system prompt automatically.

Run `/init` to generate a starter template, then edit it to define your project's coding standards, preferences, and constraints.

## How to Extend

### Add a new tool

1. Add the function schema to `TOOL_DEFINITIONS` in `tools.py`
2. Implement the function (e.g., `_my_tool(...)`)
3. Register it in `TOOL_MAP`
4. If it's a write/execute tool, add its name to `WRITE_TOOLS` in `agent.py`

### Ideas for next features

- **MCP support** — connect to external services
- **Subagents** — spawn parallel agents for subtasks
- **Hooks** — run scripts before/after tool execution
- **Session persistence** — save/resume conversations
- **Context compaction v2** — LLM-powered summarization of old context

## Environment Variables

| Variable                   | Description                       | Default             |
|----------------------------|-----------------------------------|---------------------|
| `AGENTCODE_MODEL`          | Default model                     | `claude-sonnet-4-6` |
| `AGENTCODE_MAX_ITERATIONS` | Max tool-call iterations per turn | `25`                |
| `ANTHROPIC_API_KEY`        | Anthropic API key                 | —                   |
| `OPENAI_API_KEY`           | OpenAI API key                    | —                   |

## License

MIT
