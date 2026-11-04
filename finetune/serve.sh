#!/usr/bin/env bash
# Serve the AgentCode lineup with vLLM (OpenAI-compatible API).
#
#   finetune/serve.sh axon-1            # on :8001
#   finetune/serve.sh tandav-1          # on :8002 (2 GPUs)
#   finetune/serve.sh veda-1            # on :8003
#   finetune/serve.sh axon-1 /path/to/merged   # weights from elsewhere
#
# Then point AgentCode at them (shell or workspace .env):
#   AGENTCODE_FT_AXON_1_API_BASE=http://localhost:8001/v1
#   AGENTCODE_FT_TANDAV_1_API_BASE=http://localhost:8002/v1
#   AGENTCODE_FT_VEDA_1_API_BASE=http://localhost:8003/v1
#
# The served name equals the AgentCode name, so no extra settings are needed.
# The hermes tool-call parser turns Qwen's <tool_call> output into structured
# OpenAI tool_calls — the same shape AgentCode gets from Claude/GPT.
#
# On a Mac, use the GGUF instead (train.py --gguf q4_k_m writes a Modelfile):
#   ollama create axon-1 -f finetune/out/axon-1/gguf/Modelfile
#   AGENTCODE_FT_AXON_1_API_BASE=http://localhost:11434/v1
set -euo pipefail

model="${1:?usage: serve.sh axon-1|tandav-1|veda-1 [weights_dir]}"
here="$(cd "$(dirname "$0")" && pwd)"
weights="${2:-$here/out/$model/merged}"

# tandav-1 is an 80B MoE: bf16 needs ~160GB, so it's split across 2 GPUs by
# default. On one 80GB GPU use FP8: VLLM_EXTRA_ARGS="--quantization fp8" TP=1
case "$model" in
  axon-1)   port=8001; ctx=16384; tp="${TP:-1}" ;;
  tandav-1) port=8002; ctx=32768; tp="${TP:-2}" ;;
  veda-1)   port=8003; ctx=16384; tp="${TP:-1}" ;;
  chitra-1) echo "chitra-1 is an image model; it isn't served with vLLM." >&2; exit 1 ;;
  *) echo "unknown model: $model" >&2; exit 1 ;;
esac

if [[ ! -d "$weights" ]]; then
  echo "No weights at $weights — run: python finetune/train.py $model" >&2
  exit 1
fi

exec vllm serve "$weights" \
  --served-model-name "$model" \
  --port "$port" \
  --max-model-len "$ctx" \
  --tensor-parallel-size "$tp" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  ${VLLM_EXTRA_ARGS:-}
