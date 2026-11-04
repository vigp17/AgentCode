"""
AgentCode — fine-tuned model registry.

The AgentCode lineup:

    axon-1    Qwen2.5-Coder-14B-Instruct   unified code model, low/medium/high effort tags
    tandav-1  Qwen3-Next-80B-A3B-Instruct  apex architect (80B MoE), up to 50 agent steps
    veda-1    Qwen3-14B                    deep research: docs, context, cited specs
    chitra-1  FLUX.1-schnell               image model (for /image)

The chat models are LoRA fine-tunes trained on AgentCode's own tool-calling
trajectories (see finetune/), so they speak AgentCode's tool schema natively.
They share Qwen's chat template and <tool_call> format, so one training recipe,
one vLLM tool parser and one fallback parser cover them all. Weights are served
from any OpenAI-compatible endpoint — vLLM, SGLang, Ollama (`/v1`), LM Studio,
Together, Fireworks, OpenRouter — reached through litellm's `openai/` provider.

Where each model is served is configured by (highest precedence first):
  1. settings.json  →  "finetuned": {"axon-1": {"api_base": ..., "served_name": ...}}
  2. env            →  AGENTCODE_FT_<NAME>_API_BASE, e.g. AGENTCODE_FT_AXON_1_API_BASE

The lineup always appears in the VS Code picker, marked "not connected" until
it has an endpoint. Extra fine-tunes of your own can be added in settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from model_discovery import DiscoveredModel

DEFAULT_API_BASE = "http://localhost:8000/v1"


@dataclass
class FineTunedModel:
    name: str                 # what users type: "axon-1"
    tier: str                 # routing tier: "light" | "medium" | "heavy" (or a role)
    label: str
    base_model: str           # Hugging Face id of the base it was tuned from
    served_name: str = ""     # model name the endpoint expects (defaults to name)
    api_base: str = DEFAULT_API_BASE
    api_key_env: str = "AGENTCODE_FT_API_KEY"
    # Self-hosted → $0 by default. Set if you serve on a paid endpoint so
    # /cost stays honest. $ per 1M tokens.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    configured: bool = False  # True once the user pointed it at an endpoint
    kind: str = "chat"        # "chat" | "image"
    supports_effort: bool = False   # accepts low/medium/high effort tags
    max_iterations: int | None = None  # per-model agent-loop cap (None = settings)
    description: str = ""

    @property
    def litellm_model(self) -> str:
        return f"openai/{self.served_name or self.name}"

    def completion_kwargs(self) -> dict:
        return {
            "model": self.litellm_model,
            "api_base": self.api_base,
            # OpenAI-compatible servers usually ignore the key, but litellm's
            # openai client refuses to send a request without one.
            "api_key": os.environ.get(self.api_key_env) or "agentcode-local",
        }


# The AgentCode ecosystem: four in-house models, always listed in the VS Code
# picker (marked "not connected" until an endpoint is configured with
# AGENTCODE_FT_<NAME>_API_BASE, e.g. AGENTCODE_FT_AXON_1_API_BASE, or settings).
ECOSYSTEM: list[FineTunedModel] = [
    FineTunedModel(
        name="axon-1", tier="medium", label="axon-1",
        base_model="Qwen/Qwen2.5-Coder-14B-Instruct", supports_effort=True,
        description="Unified code model: everyday coding, reads and surgical edits",
    ),
    FineTunedModel(
        name="tandav-1", tier="heavy", label="tandav-1",
        base_model="Qwen/Qwen3-Next-80B-A3B-Instruct", max_iterations=50,
        description="Apex architect: hardest multi-file work, up to 50 agent steps",
    ),
    FineTunedModel(
        name="veda-1", tier="research", label="veda-1",
        base_model="Qwen/Qwen3-14B",
        description="Deep research: docs, web context, cited specs",
    ),
    FineTunedModel(
        name="chitra-1", tier="image", label="chitra-1", kind="image",
        base_model="black-forest-labs/FLUX.1-schnell",
        description="Visual assets: icons, diagrams, images into your workspace",
    ),
]

# Effort levels for models with supports_effort. The tag is added to the
# system prompt (the fine-tune is trained on it); the iteration cap bounds
# how much work one turn may do.
EFFORT_LEVELS = ("low", "medium", "high")
EFFORT_MAX_ITERATIONS = {"low": 8, "medium": 25, "high": 40}


def effort_system_suffix(effort: str) -> str:
    return f"\n\n<effort>{effort}</effort>"


def _env_name(name: str) -> str:
    return name.upper().replace("-", "_").replace(".", "_")


_registry: dict[str, FineTunedModel] = {m.name: m for m in ECOSYSTEM}


def configure(overrides: dict | None = None) -> None:
    """Apply env vars and settings.json "finetuned" overrides to the registry.

    overrides is {name: {api_base, served_name, api_key_env, label, tier,
    base_model, input_cost_per_mtok, output_cost_per_mtok}}. New names add
    extra fine-tunes beyond the lineup.
    """
    global _registry
    reg: dict[str, FineTunedModel] = {}
    for eco in ECOSYSTEM:
        m = replace(eco)
        url = os.environ.get(f"AGENTCODE_FT_{_env_name(m.name)}_API_BASE")
        if url:
            m.api_base, m.configured = url, True
        reg[m.name] = m

    for name, cfg in (overrides or {}).items():
        if not isinstance(cfg, dict):
            continue
        base = reg.get(name) or FineTunedModel(
            name=name, tier=cfg.get("tier", "medium"), label=cfg.get("label", name),
            base_model=cfg.get("base_model", ""),
        )
        allowed = {f for f in FineTunedModel.__dataclass_fields__ if f not in ("name", "configured")}
        updates = {k: v for k, v in cfg.items() if k in allowed}
        reg[name] = replace(base, **updates, configured=True)

    _registry = reg


def get(name: str) -> FineTunedModel | None:
    return _registry.get(name)


def is_finetuned(name: str) -> bool:
    return name in _registry


def all_models(include_unconfigured: bool = False) -> list[FineTunedModel]:
    return [m for m in _registry.values() if include_unconfigured or m.configured]


def tiers() -> dict[str, str]:
    """{tier: name} for configured fine-tunes — used by the "agentcode" provider."""
    out: dict[str, str] = {}
    for m in all_models():
        out.setdefault(m.tier, m.name)
    return out


def pricing(name: str) -> tuple[float, float] | None:
    m = get(name)
    return (m.input_cost_per_mtok, m.output_cost_per_mtok) if m else None


def completion_kwargs(model: str) -> dict:
    """kwargs for litellm.completion(): resolves fine-tune names to their
    endpoint, passes every other model string through untouched."""
    m = get(model)
    return m.completion_kwargs() if m else {"model": model}


def is_ecosystem(name: str) -> bool:
    return any(m.name == name for m in ECOSYSTEM)


def as_discovered() -> list[DiscoveredModel]:
    """Ecosystem models always (so the picker shows the lineup), plus any
    other fine-tune that has an endpoint."""
    eco_names = [m.name for m in ECOSYSTEM]
    rows = [_registry[n] for n in eco_names if n in _registry]
    rows += [m for m in all_models() if m.name not in eco_names]
    return [
        DiscoveredModel(
            id=m.name, provider="agentcode", label=m.label, source="finetuned",
            kind=m.kind, available=m.configured, supports_effort=m.supports_effort,
            description=m.description,
        )
        for m in rows
    ]
