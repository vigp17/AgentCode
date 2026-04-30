# AgentCode

An open, multi-model agentic coding CLI — inspired by Claude Code.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  cli.py (UI)                    │
│  REPL loop · slash commands · Rich terminal UI  │
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

| Tool             | Description                          | Permission |
|------------------|--------------------------------------|------------|
| `read_file`      | Read file contents with line numbers | Auto       |
| `write_file`     | Create or overwrite a file           | Ask        |
| `edit_file`      | Surgical find-and-replace edit       | Ask        |
| `run_command`    | Execute a bash command               | Ask        |
| `list_directory` | Tree view of directory structure     | Auto       |
| `search_files`   | Find files by glob pattern           | Auto       |
| `search_text`    | Grep for text across files           | Auto       |

**Permission model:** Read-only tools auto-approve. Write/execute tools ask before running (unless `--auto-approve` / `-y` flag is set).

## Slash Commands

| Command           | Description                    |
|-------------------|--------------------------------|
| `/model <name>`   | Switch LLM model on the fly    |
| `/clear`          | Reset conversation history     |
| `/compact`        | Force context window compaction|
| `/tokens`         | Show estimated token usage     |
| `/help`           | Show help                      |
| `/exit`           | Quit                           |

## How to Extend

### Add a new tool

1. Add the function schema to `TOOL_DEFINITIONS` in `tools.py`
2. Implement the function (e.g., `_my_tool(...)`)
3. Register it in `TOOL_MAP`

### Ideas for next features

- **AGENTCODE.md** — project-level instructions file (like CLAUDE.md)
- **Git integration** — auto-commit, branch management, PR creation
- **MCP support** — connect to external services
- **Subagents** — spawn parallel agents for subtasks
- **Hooks** — run scripts before/after tool execution
- **Session persistence** — save/resume conversations
- **Context compaction v2** — LLM-powered summarization of old context

## Environment Variables

| Variable                   | Description                       | Default                     |
|----------------------------|-----------------------------------|-----------------------------|
| `AGENTCODE_MODEL`          | Default model                     | `claude-sonnet-4-6`         |
| `AGENTCODE_MAX_ITERATIONS` | Max tool-call iterations per turn | `25`                        |
| `ANTHROPIC_API_KEY`        | Anthropic API key                 | —                           |
| `OPENAI_API_KEY`           | OpenAI API key                    | —                           |

## License

MIT
