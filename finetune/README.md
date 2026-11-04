# AgentCode fine-tuned models

AgentCode has three models of its own, one per routing tier. They are LoRA
fine-tunes of the Qwen3 family, trained on AgentCode's own tool-calling
trajectories so they use AgentCode's tools (`read_file`, `edit_file`,
`run_command`, …) natively and can run on your own hardware at $0 per token.

| Model | Base | Size | Learns from | Train on | Serve on |
|-------|------|------|-------------|----------|----------|
| `agentcode-light`  | Qwen3-4B-Instruct-2507      | 4B dense | short trajectories (≤3 tool calls) | 1× 24GB GPU | 1× 16GB GPU, or laptop/CPU via GGUF |
| `agentcode-medium` | Qwen3-14B                   | 14B dense | ≤10 tool calls | 1× 48GB GPU | 1× 48GB, or 24GB with FP8/AWQ |
| `agentcode-heavy`  | Qwen3-Next-80B-A3B-Instruct | 80B MoE, 3B active | everything, incl. long multi-file work | 1× 80GB GPU (QLoRA) | 2× 80GB (bf16) or 1× 80GB (FP8) |

All three use the same chat template and `<tool_call>` format, so one recipe,
one vLLM tool parser and AgentCode's built-in fallback parser cover them all.
Heavy is a mixture-of-experts model: it has 80B parameters of knowledge but
activates only about 3B per token, so it serves much faster than an 80B dense
model. Its LoRA targets attention and the shared expert, not all 512 routed
experts. Check the module names with `print(model)` before a long run.

**Other sizes.** For an exact 12B medium, set `base_model` to
`mistralai/Mistral-Nemo-Instruct-2407`. That model uses a different chat
template, so also update the markers at the top of `train.py`. For a heavy in
the 100–120B range, `openai/gpt-oss-120b` (117B MoE) and `zai-org/GLM-4.5-Air`
(106B MoE) are strong agentic bases. They also use their own templates and tool
formats, so serve them with the matching vLLM `--tool-call-parser`.

Hyperparameters live in [`configs/`](configs/). To use a different base,
edit `base_model` there or pass `--base` to `train.py`.

> This directory holds the pipeline. The weights aren't shipped in this repo:
> you train them with your own data and GPU (or with OpenAI's hosted
> fine-tuning, see below).

## Pipeline

```
 collect ──► prepare ──► train ──► serve ──► connect ──► evaluate
```

### 1. Collect trajectories

**From your own use.** Turn on logging in `.agentcode/settings.json` (or the
global `~/.agentcode/settings.json`) and use AgentCode normally:

```json
{ "training": { "log_trajectories": true } }
```

Every turn that ends in a final answer is appended to
`~/.agentcode/trajectories/YYYY-MM-DD.jsonl`. Turns that hit the iteration cap
or errored are skipped. These files contain your prompts and code, so the
setting is off by default.

**Bootstrapped.** Run AgentCode headless over a list of repos and tasks:

```bash
python finetune/generate_data.py --tasks finetune/tasks.example.jsonl \
    --teacher <litellm model string> --out finetune/raw
```

This runs agent-written commands with auto-approve, so do it in a container or VM.

> **Choosing a teacher model.** Anthropic's, OpenAI's and Google's terms
> generally restrict using their models' outputs to train other models. That
> covers trajectories logged while you use Claude, GPT or Gemini too. Use an
> open-weight teacher whose license allows distillation (for example a large
> Qwen-Coder or DeepSeek model served by Together, Fireworks or your own vLLM),
> or check your agreement with the provider first.

### 2. Prepare

```bash
python finetune/prepare_data.py                       # reads ~/.agentcode/trajectories
python finetune/prepare_data.py --input finetune/raw  # or any dirs/files
```

This validates every tool call (known tool, JSON arguments, every call has a
result), truncates huge tool outputs, removes duplicates and strips MCP tools.
It then writes a per-tier slice to `finetune/data/agentcode-<tier>/{train,eval}.jsonl`.
The eval split is held out **by project**, so scores measure generalization.

### 3. Train

```bash
pip install -r finetune/requirements.txt   # CUDA GPU required
python finetune/train.py light
python finetune/train.py medium
python finetune/train.py heavy
python finetune/train.py light --gguf q4_k_m   # also export for Ollama / llama.cpp
python finetune/train.py light --max-steps 20  # smoke test
```

`train.py` uses Unsloth when it supports the model. Otherwise, for example if
your Unsloth version doesn't support Qwen3-Next yet, it falls back to plain
transformers + PEFT with 4-bit QLoRA and the same assistant-only loss. Force
either with `--trainer unsloth` or `--trainer hf`. The fallback spreads the
model across all visible GPUs (`device_map="auto"`).

Loss is computed on assistant turns only; system prompts, user messages and
tool results are context. Output goes to `finetune/out/agentcode-<tier>/`:
`lora/` (adapter) and `merged/` (16-bit weights for serving).

**No GPU?** Use OpenAI's hosted fine-tuning on the same data:

```bash
python finetune/train.py medium --backend openai --base <fine-tunable OpenAI model>
```

When the job finishes, the `ft:…` model shows up in AgentCode's model list on
its own, because model discovery lists your account's fine-tunes. Run
`/models refresh` or click ↻ in the VS Code picker.

### 4. Serve

```bash
finetune/serve.sh light     # vLLM on :8001
finetune/serve.sh medium    # vLLM on :8002
finetune/serve.sh heavy     # vLLM on :8003, split across 2 GPUs
TP=1 VLLM_EXTRA_ARGS="--quantization fp8" finetune/serve.sh heavy   # one 80GB GPU
```

Any OpenAI-compatible server works: vLLM, SGLang, Ollama (`http://localhost:11434/v1`),
LM Studio, or a hosted LoRA endpoint on Together or Fireworks. With vLLM,
use `--tool-call-parser hermes` (serve.sh sets it). AgentCode can also parse
`<tool_call>` text itself for servers that don't.

### 5. Connect AgentCode

In `.env` or your shell:

```bash
AGENTCODE_FT_LIGHT_API_BASE=http://localhost:8001/v1
AGENTCODE_FT_MEDIUM_API_BASE=http://localhost:8002/v1
AGENTCODE_FT_HEAVY_API_BASE=http://localhost:8003/v1
# AGENTCODE_FT_API_BASE=...   one endpoint for all three
# AGENTCODE_FT_API_KEY=...    if your endpoint needs one
```

or in `settings.json`, which also lets you add more fine-tunes of your own:

```json
{
  "finetuned": {
    "agentcode-medium": { "api_base": "http://gpu-box:8002/v1" },
    "agentcode-heavy":  { "api_base": "https://api.together.xyz/v1",
                          "served_name": "you/agentcode-heavy-lora",
                          "api_key_env": "TOGETHER_API_KEY",
                          "input_cost_per_mtok": 0.8, "output_cost_per_mtok": 0.8 }
  }
}
```

A fine-tune only appears in the picker once it has an endpoint. Then:

```bash
agentcode --model agentcode-medium            # one model
agentcode --model agentcode-light             # routing on: light/medium/heavy fine-tunes
```

When you pick any `agentcode-*` model with routing on, AgentCode routes
between whichever of the three fine-tunes you've configured.

### 6. Evaluate

```bash
python finetune/eval.py agentcode-medium --baseline claude-haiku-4-5
```

The eval gives the model each held-out conversation up to an assistant step
and checks whether its next action matches the reference: the same tool, or a
final answer. It reports `action_acc`, `valid_call` (real tool, JSON
arguments) and `args_match`.
