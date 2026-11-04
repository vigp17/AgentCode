# Training the AgentCode lineup

This folder turns AgentCode's own tool-calling trajectories into the models in
the VS Code picker. Once a model is trained and served, its "not connected"
badge goes away.

| Model | Base | Size | Learns from | Train on | Serve on |
|-------|------|------|-------------|----------|----------|
| `axon-1`   | Qwen2.5-Coder-14B-Instruct  | 14B dense | everyday trajectories (≤40 tool calls), each tagged low / medium / high effort | 1× 24GB GPU, or free Colab/Kaggle T4 | 1× 24GB GPU, or a 32GB Mac via Ollama |
| `tandav-1` | Qwen3-Next-80B-A3B-Instruct | 80B MoE, 3B active | everything with real work in it, including the longest | 2–8× 80GB GPUs (DeepSpeed ZeRO-2) | 2× 80GB (bf16) or 1× 80GB (FP8) |
| `veda-1`   | Qwen3-14B                   | 14B dense | research trajectories: read/search-only, ending in a long cited answer | 1× 24GB GPU, or free Colab/Kaggle T4 | 1× 24GB GPU |
| `chitra-1` | FLUX.1-schnell              | image | — (see [below](#chitra-1)) | | |

All three chat models use Qwen's chat template and `<tool_call>` format, so one
recipe, one vLLM tool parser and AgentCode's fallback parser cover them all.
Hyperparameters live in [`configs/`](configs/); edit `base_model` there to use
a different base.

> **veda-1's base differs from the plan's example.** DeepSeek-R1-Distill's chat
> template has no tool calling, and veda-1 has to call read and search tools.
> Qwen3-14B is a reasoning model with native tool calls and the same template
> as the others. The config's `_base_model_note` explains how to switch back.

## Pipeline

```
 collect ──► prepare (score + slice) ──► train ──► serve ──► connect ──► evaluate
```

### 1. Collect trajectories

**From your own use.** Turn on logging in `.agentcode/settings.json` (or
`~/.agentcode/settings.json`) and use AgentCode normally:

```json
{ "training": { "log_trajectories": true } }
```

Every turn that ends in a final answer is appended to
`~/.agentcode/trajectories/YYYY-MM-DD.jsonl`. This is off by default because
the files contain your prompts and code.

**Bootstrapped.** `tasks.jsonl` has **398 ready-made tasks** across 50 real
open-source repos (Python, JS/TS, Go and Rust):

| Category | Tasks | Mostly trains |
|----------|------:|---------------|
| coding (low/medium effort) | 220 | axon-1 |
| research (read-only, cited write-ups) | 127 | veda-1 |
| long (multi-file, many steps) | 51 | tandav-1, axon-1 "high" |

Do a trial on 5 tasks first, then run the whole set:

```bash
python finetune/generate_data.py --tasks finetune/tasks.jsonl \
    --teacher <litellm model string> --out finetune/raw --limit 5

python finetune/generate_data.py --tasks finetune/tasks.jsonl \
    --teacher <litellm model string> --out finetune/raw --workers 4
```

Finished tasks are recorded in `finetune/raw/done.txt`, so re-running the same
command resumes where it stopped. Each task runs in a fresh copy of its repo
with commands auto-approved, so **run this in a container or VM**. The agent
installs each repo's dependencies itself. Expect a few minutes per task
(roughly 2–4 hours for the full set at `--workers 4`). Teacher cost depends on
its per-token price; at ~100k tokens per task, budget for ~40M tokens in total.

To add repos or tasks, edit `make_tasks.py` (repo list, template pools and
repo-specific tasks), then run `python finetune/make_tasks.py`.
`tasks.example.jsonl` is a 13-task sample.

> **Choosing a teacher model.** Anthropic's, OpenAI's and Google's terms
> generally restrict using their models' outputs to train other models. That
> covers trajectories logged while you use Claude, GPT or Gemini too. Use an
> open-weight teacher whose license allows distillation (for example a large
> Qwen3-Coder or DeepSeek model via Together, Fireworks, OpenRouter or your own
> vLLM), or check your agreement with the provider first.

### 2. Prepare: score, filter, slice

```bash
python finetune/prepare_data.py --input finetune/raw --top 500
python finetune/prepare_data.py --model axon-1 --min-reward 0.6
```

Every trajectory is validated: the tool exists, the arguments are JSON, and
every call has a result. Each one then gets a **reward score**:

| | |
|---|---|
| +0.5 | valid and ends in an answer (base) |
| +0.2 | ran tests, linters or type checks after its last edit |
| +0.1 | substantive final answer |
| −0.1 | per tool error hit (max −0.3) |
| −0.2 | looping: the same call issued 3+ times |

Trajectories under `--min-reward` (default 0.5) are dropped, the rest are
ranked best first, and `--top N` keeps the best N per model. The heuristic is
in `reward()` in `prepare_data.py` so you can tune it.

Each model then gets its slice in `finetune/data/<model>/{train,eval}.jsonl`.
The eval split is held out **by project**, so scores measure generalization.
For axon-1, each example is tagged `<effort>low|medium|high</effort>` from how
much work it took (≤2 tool calls low, ≤8 medium, more is high). It's the exact
tag the VS Code effort pills send at inference, so axon-1 learns what each
level means.

### 3. Train

```bash
pip install -r finetune/requirements.txt     # CUDA GPU required

python finetune/train.py axon-1
python finetune/train.py veda-1
python finetune/train.py axon-1 --gguf q4_k_m            # + GGUF and Ollama Modelfile
python finetune/train.py axon-1 --push-to-hub you/axon-1 # + upload (needs HF_TOKEN)
python finetune/train.py axon-1 --max-steps 20           # smoke test first
```

Loss is computed on assistant turns only; system prompts, user messages and
tool results are context. Output goes to `finetune/out/<model>/`: `lora/`
(the adapter), `merged/` (16-bit weights for vLLM), and `gguf/` (if you asked
for it).

**Colab / Kaggle (axon-1, veda-1).** Open
[`colab_train.ipynb`](colab_train.ipynb) and run it top to bottom. It clones
the repo, reads your trajectories from Google Drive, trains with `--colab`
(4-bit, 4k context, fits a free T4) and saves the GGUF back to Drive. The
`--colab` preset skips the ~28GB merged weights. If GGUF export runs out of
RAM on free Colab, use Kaggle (more RAM), or push the adapter with
`--push-to-hub` and convert it later.

**tandav-1 (80B MoE).** It needs real GPUs: rent 2–8× H100/A100, or use a
research allocation. It trains with r=128 adapters and DeepSpeed ZeRO-2
(`ds_zero2.json`):

```bash
pip install deepspeed
accelerate launch --num_processes 4 finetune/train.py tandav-1
```

On a single 80GB GPU it also runs without DeepSpeed, at shorter context:
lower `max_seq_length` in `configs/tandav-1.json` to 8192. The adapter targets
attention and the shared expert, not all 512 routed experts. Check the module
names with `print(model)` before a long run. Unsloth doesn't do multi-GPU, so
this always uses the transformers + PEFT trainer.

**No GPU at all?** `--backend openai --base <fine-tunable OpenAI model>` trains
on OpenAI's servers instead. The `ft:` model appears in the picker
automatically, but under OpenAI's name rather than as axon-1.

**Not included yet:** GRPO / RL. The plan has it for axon-1 and veda-1, but
it should come after a working SFT model and a reward that runs code (tests
pass), not just the heuristic above. This pipeline is the SFT stage.

### 4. Serve

**Mac (Ollama, from the GGUF):**

```bash
ollama create axon-1 -f finetune/out/axon-1/gguf/Modelfile
```

**GPU server (vLLM, from `merged/`):**

```bash
finetune/serve.sh axon-1      # :8001
finetune/serve.sh tandav-1    # :8002, split across 2 GPUs
finetune/serve.sh veda-1      # :8003
TP=1 VLLM_EXTRA_ARGS="--quantization fp8" finetune/serve.sh tandav-1   # one 80GB GPU
```

Both serve the model under its AgentCode name, so no `served_name` is needed.
Hosted LoRA endpoints work too (Together, Fireworks).

### 5. Connect AgentCode

In your workspace `.env`:

```bash
AGENTCODE_FT_AXON_1_API_BASE=http://localhost:11434/v1     # Ollama
AGENTCODE_FT_TANDAV_1_API_BASE=http://gpu-box:8002/v1      # vLLM
AGENTCODE_FT_VEDA_1_API_BASE=http://gpu-box:8003/v1
# AGENTCODE_FT_API_KEY=...   if your endpoint needs one
```

or in `.agentcode/settings.json`. This also takes a hosted endpoint whose
model name differs from the AgentCode name:

```json
{
  "finetuned": {
    "tandav-1": { "api_base": "https://api.together.xyz/v1",
                  "served_name": "you/tandav-1-lora",
                  "api_key_env": "TOGETHER_API_KEY",
                  "input_cost_per_mtok": 0.9, "output_cost_per_mtok": 0.9 }
  }
}
```

Reload VS Code. The dot turns green and the model answers in the panel.

### 6. Evaluate

```bash
python finetune/eval.py axon-1 --baseline claude-haiku-4-5
```

The eval gives the model each held-out conversation up to an assistant step
and checks whether its next action matches the reference: the same tool, or a
final answer. It reports `action_acc`, `valid_call` (real tool, JSON
arguments) and `args_match`. Run it against the untrained base too, so you can
see what fine-tuning bought.

## chitra-1

chitra-1 is an image model, so none of the above applies: it doesn't use
trajectories, tools or a chat template. You can use FLUX.1-schnell as-is to
start (Apache 2.0). To give it a house style for icons and diagrams, train a
style LoRA on 20–50 example images with a FLUX LoRA trainer such as
[ai-toolkit](https://github.com/ostris/ai-toolkit) or diffusers'
`train_dreambooth_lora_flux.py`. Wiring it into AgentCode is the `/image`
command, which isn't built yet.
