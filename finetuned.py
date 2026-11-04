"""
AgentCode — fine-tuned model registry.

AgentCode ships three of its own fine-tuned models, one per routing tier:

    agentcode-light   Qwen3-4B-Instruct-2507        (4B dense)       quick answers, cheap edits
    agentcode-medium  Qwen3-14B                     (14B dense)      everyday coding tasks
    agentcode-heavy   Qwen3-Next-80B-A3B-Instruct   (80B MoE, 3B active)  multi-file / long-horizon work

All three share Qwen's chat template and <tool_call> format, so one training
recipe, one vLLM tool parser and one fallback parser cover them all.

They are trained on AgentCode's own tool-calling trajectories (see finetune/),
so they speak AgentCode's tool schema natively. Weights are served from any
OpenAI-compatible endpoint — vLLM, SGLang, Ollama (`/v1`), LM Studio, Together,
Fireworks — and AgentCode reaches them through litellm's `openai/` provider.

Where each model is served is configured, in order of precedence, by:
  1. settings.json  →  "finetuned": {"agentcode-medium": {"api_base": ...}}
  2. env            →  AGENTCODE_FT_<TIER>_API_BASE / AGENTCODE_FT_API_BASE
  3. default        →  http://localhost:8000/v1   (vLLM's default port)

A fine-tune is only offered in the picker and routing once it has an endpoint
configured (any of the above other than the default), or settings explicitly
list it — so users who never trained one don't see dead entries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from model_discovery import DiscoveredModel

DEFAULT_API_BASE = "http://localhost:8000/v1"


@dataclass
class FineTunedModel:
    name: str                 # what users type: "agentcode-medium"
    tier: str                 # "light" | "medium" | "heavy"
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


BUILTIN_FINETUNES: list[FineTunedModel] = [
    FineTunedModel(
        name="agentcode-light", tier="light", label="AgentCode Light (4B)",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    ),
    FineTunedModel(
        name="agentcode-medium", tier="medium", label="AgentCode Medium (14B)",
        base_model="Qwen/Qwen3-14B",
    ),
    FineTunedModel(
        name="agentcode-heavy", tier="heavy", label="AgentCode Heavy (80B MoE)",
        base_model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    ),
]

_registry: dict[str, FineTunedModel] = {m.name: m for m in BUILTIN_FINETUNES}


def configure(overrides: dict | None = None) -> None:
    """Apply env vars and settings.json "finetuned" overrides to the registry.

    overrides is {name: {api_base, served_name, api_key_env, label, tier,
    base_model, input_cost_per_mtok, output_cost_per_mtok}}. New names add
    extra fine-tunes beyond the built-in three.
    """
    global _registry
    reg = {m.name: replace(m) for m in BUILTIN_FINETUNES}

    shared = os.environ.get("AGENTCODE_FT_API_BASE")
    for m in reg.values():
        per_tier = os.environ.get(f"AGENTCODE_FT_{m.tier.upper()}_API_BASE")
        if per_tier or shared:
            m.api_base = per_tier or shared
            m.configured = True

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


def as_discovered() -> list[DiscoveredModel]:
    return [
        DiscoveredModel(id=m.name, provider="agentcode", label=m.label, source="finetuned")
        for m in all_models()
    ]
