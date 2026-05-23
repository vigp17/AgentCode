# AgentCode

**Agentic AI coding assistant for VS Code** — chat, edit, run tools, and get inline completions powered by Claude, GPT-4o, or Gemini. Open source, multi-model, and cost-aware.

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
| `agentcode.azureApiKey` | Azure OpenAI key |
| `agentcode.azureEndpoint` | `https://my-resource.openai.azure.com` |

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
Ghost-text suggestions appear as you type, powered by Claude Haiku. Press `Tab` to accept. Toggle on/off anytime via `AgentCode: Toggle inline completions` in the Command Palette.

### Multi-Model Support
Switch models mid-session from the dropdown in the chat panel. Works with:

| Provider | Models |
|----------|--------|
| Anthropic | `claude-sonnet-4-6` (default), `claude-haiku-4-5`, `claude-opus-4-7` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Google | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` |
| Azure OpenAI | `azure/<your-deployment>` (e.g. `azure/gpt-4o`) |

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
| `agentcode.azureApiKey` | — | Azure OpenAI API key |
| `agentcode.azureEndpoint` | — | Azure OpenAI endpoint URL |
| `agentcode.model` | `claude-sonnet-4-6` | Default model |
| `agentcode.executablePath` | `agentcode` | Path to `agentcode` binary if not on PATH |
| `agentcode.inlineCompletions.enabled` | `true` | Enable inline completions |

Keys can also be set via environment variables or a `.env` file in your project root — VS Code settings take priority.

---

## Links

- [GitHub](https://github.com/vigp17/AgentCode) — source code, issues, contributions welcome
- [PyPI](https://pypi.org/project/agentcode-cli/) — CLI package

## License

MIT
