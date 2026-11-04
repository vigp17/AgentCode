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

### The AgentCode lineup

The model dropdown always lists AgentCode's own models:

| Model | Role |
|-------|------|
| `axon-1` | Unified code model for everyday coding, file reads and surgical edits. Has a **Reasoning Effort** selector (Low / Medium / High) under the input |
| `tandav-1` | Apex architect for the hardest multi-file work, up to 50 agent steps |
| `veda-1` | Deep research: docs, web context, cited specs |
| `chitra-1` | Visual assets (image model; `/image` is coming) |

A model shows **not connected** until you serve it and set its endpoint, for
example `AGENTCODE_FT_AXON_1_API_BASE=http://localhost:8002/v1` in your
workspace `.env`.

**Keyboard:** type `/` in the input for commands. `/model ax` autocompletes
model names (↑/↓ to choose, Tab or Enter to accept), `/effort high` sets the
effort, and `/clear` starts a new chat. Switching model or effort takes effect
immediately and keeps the conversation; the backend isn't restarted.

### Multi-Model Support
Switch models mid-session from the dropdown in the chat panel. **The dropdown fills itself.** The backend asks Anthropic, OpenAI and Google which models your API keys can use, so a newly released model appears without an extension or CLI update. Use the filter box to search, or click **↻ Check for new models** to re-check right away.

The dropdown also lists:

- **Your OpenAI fine-tunes** (`ft:…` models), picked up from your account automatically.
- **AgentCode's own fine-tuned models** (`agentcode-light`, `agentcode-medium`, `agentcode-heavy`), once you've pointed AgentCode at a server running them (for example `AGENTCODE_FT_MEDIUM_API_BASE=http://localhost:8002/v1` in your workspace `.env`). See [finetune/README.md](https://github.com/vigp17/AgentCode/blob/main/finetune/README.md).

Any [LiteLLM model name](https://docs.litellm.ai/docs/providers) also works in the `agentcode.model` setting.

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
