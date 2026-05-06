# AgentCode

AI coding assistant — multi-model, cost-aware, open source. Inspired by Claude Code.

## Requirements

Install the CLI before using the extension:

```bash
pip install agentcode-cli
```

## Setup

Add your API keys in VS Code Settings (`Cmd+,` → search **AgentCode**):

| Setting | Description |
|---------|-------------|
| `agentcode.anthropicApiKey` | Anthropic API key (`sk-ant-...`) |
| `agentcode.openaiApiKey` | OpenAI API key (`sk-...`) |
| `agentcode.geminiApiKey` | Google Gemini API key |
| `agentcode.model` | Default model (e.g. `claude-sonnet-4-6`) |
| `agentcode.executablePath` | Path to `agentcode` if not on PATH |

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `AgentCode: Open` | `Cmd+Shift+A` | Open the chat panel |
| `AgentCode: Ask about selection` | Right-click | Ask about highlighted code |
| `AgentCode: Explain this file` | Right-click | Explain the current file |

## Features

- **Multi-model** — works with Claude, GPT, and Gemini via a single interface
- **Cost-aware routing** — automatically picks the cheapest model that can handle the task
- **Live streaming** — responses stream in real time in the side panel
- **Active file context** — your current file is sent automatically, no copy-pasting needed
- **Diff viewer** — write/edit tool calls open VS Code's native diff viewer for review before applying
- **Subagents** — spawns parallel agents for independent subtasks

## Supported Models

| Provider | Model |
|----------|-------|
| Anthropic | `claude-sonnet-4-6` (default), `claude-haiku-4-5`, `claude-opus-4-6` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Google | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` |

## Links

- [GitHub](https://github.com/vigp17/AgentCode)
- [PyPI](https://pypi.org/project/agentcode-cli/)

## License

MIT
