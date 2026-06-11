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

By default completions use **Claude Haiku** (requires `agentcode.anthropicApiKey`). You can swap in any OpenAI-compatible endpoint — e.g. a local Ollama running an open-weight coding model — by setting `agentcode.inlineCompletions.endpoint` and `agentcode.inlineCompletions.model`.

**Example: run a small coding model locally with Ollama**

```bash
ollama pull qwen2.5-coder:3b
ollama serve
```

Then in VS Code Settings:
- `agentcode.inlineCompletions.endpoint` = `http://localhost:11434/v1`
- `agentcode.inlineCompletions.model` = `qwen2.5-coder:3b`
- Leave `agentcode.inlineCompletions.apiKey` blank

Completions are now free, local, and private.

### Multi-Model Support
Switch models mid-session from the dropdown in the chat panel. Works with:

| Provider | Models |
|----------|--------|
| Anthropic | `claude-sonnet-4-6` (default), `claude-haiku-4-5`, `claude-opus-4-7` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Google | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` |
| Local | `ollama/agentcode-27b` (the AgentCode fine-tune — see below) |

### Run the AgentCode 27B Locally (Free, Private)

AgentCode ships its own fine-tuned agentic model — a 27B Qwen-3.5 fine-tune that
emits tool calls natively. Run it with [Ollama](https://ollama.com) and pay nothing.

**Requirements:** ~20 GB free RAM (Q4_K_M quantization). A 32 GB+ Apple Silicon
Mac or any machine with a 16 GB+ GPU is comfortable.

```bash
# 1. Pull the GGUF from HuggingFace (~16 GB, one time)
ollama pull hf.co/Vigp17/agentcode-27b-gguf:Q4_K_M

# 2. Alias it to a clean name AgentCode expects
ollama cp hf.co/Vigp17/agentcode-27b-gguf:Q4_K_M agentcode-27b
```

Then pick **Local → AgentCode 27B** from the chat panel dropdown (or set
`agentcode.model` to `ollama/agentcode-27b`). AgentCode parses the model's
XML-style tool calls and strips its `<think>` reasoning automatically.

> Don't have the hardware? Deploy the model on a
> [HuggingFace Inference Endpoint](https://huggingface.co/Vigp17/agentcode-27b)
> (GPU, ~$0.50/hr, scale-to-zero) and point `agentcode.model` at it via litellm,
> or just use Claude/GPT/Gemini instead.

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
| `agentcode.model` | `claude-sonnet-4-6` | Default model |
| `agentcode.executablePath` | `agentcode` | Path to `agentcode` binary if not on PATH |
| `agentcode.inlineCompletions.enabled` | `true` | Enable inline completions |
| `agentcode.inlineCompletions.endpoint` | — | Optional OpenAI-compatible `/v1` endpoint (e.g. `http://localhost:11434/v1` for Ollama) |
| `agentcode.inlineCompletions.model` | — | Model name for the custom endpoint (e.g. `qwen2.5-coder:3b`) |
| `agentcode.inlineCompletions.apiKey` | — | API key for the custom endpoint (blank for local Ollama) |

Keys can also be set via environment variables or a `.env` file in your project root — VS Code settings take priority.

---

## Links

- [GitHub](https://github.com/vigp17/AgentCode) — source code, issues, contributions welcome
- [PyPI](https://pypi.org/project/agentcode-cli/) — CLI package

## License

MIT
