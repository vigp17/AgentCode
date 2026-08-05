# AgentCode

**Agentic AI coding assistant for VS Code** — chat, edit, run tools, and get inline completions powered by Claude, GPT, or Gemini. Open source, multi-model, and cost-aware.

---

## Quick Start

**1. Install the CLI**

```bash
pip install agentcode-cli
```

**2. Add your API key**

Open VS Code Settings (`Cmd+,` / `Ctrl+,`) and search **AgentCode**, then paste your key:

| Setting | Key |
|---------|-----|
| `agentcode.anthropicApiKey` | `sk-ant-...` |
| `agentcode.openaiApiKey` | `sk-...` |
| `agentcode.geminiApiKey` | Google AI key |

Or add a `.env` file to your project root — AgentCode reads it automatically:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Open the chat panel**

Press `Cmd+Shift+A` (Mac) / `Ctrl+Shift+A` (Windows/Linux) or run `AgentCode: Open` from the Command Palette.

---

## Features

### Agentic Chat Panel
Ask AgentCode to read files, write code, run shell commands, search your codebase, and more — all from the side panel. It streams responses in real time, shows every tool call it makes, and asks for your approval before writing or running anything.

### Inline Completions
Ghost-text suggestions appear as you type. Press `Tab` to accept. Toggle on/off anytime via `AgentCode: Toggle inline completions` in the Command Palette.

By default completions use **Claude Haiku** (requires `agentcode.anthropicApiKey`). You can swap in any OpenAI-compatible `/v1/chat/completions` endpoint — a vLLM or TGI server, or a HuggingFace Inference Endpoint — by setting `agentcode.inlineCompletions.endpoint`, `agentcode.inlineCompletions.model`, and (if the endpoint requires one) `agentcode.inlineCompletions.apiKey`.

### Multi-Model Support
Switch models mid-session from the dropdown in the chat panel. Works with:

| Provider | Models |
|----------|--------|
| Anthropic | `claude-sonnet-5` (default), `claude-haiku-4-5`, `claude-opus-5`, `claude-fable-5` |
| OpenAI | `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` |
| Google | `gemini/gemini-3.1-pro-preview`, `gemini/gemini-3.6-flash`, `gemini/gemini-3.5-flash-lite` |

The dropdown lists common choices, but any of [LiteLLM's ~3,000 model names](https://docs.litellm.ai/docs/providers) works in the `agentcode.model` setting. Newly released models become available by upgrading the CLI backend's litellm (`pip install -U litellm`) — no extension update needed.

### Cost-Aware Routing
Automatically routes each request to the cheapest model capable of handling it — heavy tasks go to powerful models, simple ones go to fast/cheap ones.

### Active File Context
Your current file is attached to every message automatically. No copy-pasting needed.

### Diff Viewer
When AgentCode wants to edit a file, VS Code's native diff viewer opens so you can review the change before it's applied.

---

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `AgentCode: Open` | `Cmd+Shift+A` | Open the chat panel |
| `AgentCode: Ask about selection` | Right-click menu | Ask about highlighted code |
| `AgentCode: Explain this file` | Right-click menu | Explain the current file |
| `AgentCode: Toggle inline completions` | Command Palette | Enable or disable ghost-text completions |

---

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `agentcode.anthropicApiKey` | — | Anthropic API key |
| `agentcode.openaiApiKey` | — | OpenAI API key |
| `agentcode.geminiApiKey` | — | Google Gemini API key |
| `agentcode.model` | `claude-sonnet-5` | Default model |
| `agentcode.executablePath` | `agentcode` | Path to `agentcode` binary if not on PATH |
| `agentcode.inlineCompletions.enabled` | `true` | Enable inline completions |
| `agentcode.inlineCompletions.endpoint` | — | Optional OpenAI-compatible `/v1` endpoint (e.g. a vLLM server or HuggingFace Inference Endpoint) |
| `agentcode.inlineCompletions.model` | — | Model name to request from the custom endpoint |
| `agentcode.inlineCompletions.apiKey` | — | API key for the custom endpoint, if it requires one |

Keys can also be set via environment variables or a `.env` file in your project root — VS Code settings take priority.

---

## Links

- [GitHub](https://github.com/vigp17/AgentCode) — source code, issues, contributions welcome
- [PyPI](https://pypi.org/project/agentcode-cli/) — CLI package

## License

MIT
