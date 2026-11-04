#!/usr/bin/env bash
# Serve AgentCode's fine-tuned models with vLLM (OpenAI-compatible API).
#
#   finetune/serve.sh light            # agentcode-light  on :8001
#   finetune/serve.sh medium           # agentcode-medium on :8002
#   finetune/serve.sh heavy            # agentcode-heavy  on :8003
#   finetune/serve.sh medium /path/to/merged   # serve weights from elsewhere
#
# Then point AgentCode at them (shell or .env):
#   AGENTCODE_FT_LIGHT_API_BASE=http://localhost:8001/v1
#   AGENTCODE_FT_MEDIUM_API_BASE=http://localhost:8002/v1
#   AGENTCODE_FT_HEAVY_API_BASE=http://localhost:8003/v1
#
# The hermes tool-call parser turns Qwen's <tool_call> output into structured
# OpenAI tool_calls, so AgentCode gets the same shape it gets from Claude/GPT.
set -euo pipefail

tier="${1:?usage: serve.sh light|medium|heavy [weights_dir]}"
here="$(cd "$(dirname "$0")" && pwd)"
weights="${2:-$here/out/agentcode-$tier/merged}"

# Heavy is an 80B MoE: bf16 needs ~160GB, so it's split across 2 GPUs by
# default. On one 80GB GPU use FP8: VLLM_EXTRA_ARGS="--quantization fp8" TP=1
case "$tier" in
  light)  port=8001; ctx=8192;  tp="${TP:-1}" ;;
  medium) port=8002; ctx=16384; tp="${TP:-1}" ;;
  heavy)  port=8003; ctx=32768; tp="${TP:-2}" ;;
  *) echo "unknown tier: $tier" >&2; exit 1 ;;
esac

if [[ ! -d "$weights" ]]; then
  echo "No weights at $weights — run: python finetune/train.py $tier" >&2
  exit 1
fi

exec vllm serve "$weights" \
  --served-model-name "agentcode-$tier" \
  --port "$port" \
  --max-model-len "$ctx" \
  --tensor-parallel-size "$tp" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  ${VLLM_EXTRA_ARGS:-}
