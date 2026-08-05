"""
AgentCode - Cost-aware model router.

Automatically picks the cheapest model that can handle the task.
Simple questions get a fast, cheap model. Complex multi-file refactors
get a powerful, expensive one.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rich.console import Console

console = Console()


# ── Model Tiers ───────────────────────────────────────────────────────────────

@dataclass
class ModelTier:
    """A model assigned to a routing tier.

    The costs here are only a fallback. Live pricing comes from litellm's
    model registry (see model_pricing), so it stays correct across price
    changes without editing this file.
    """
    name: str                    # LiteLLM model string
    tier: str                    # "light", "medium", "heavy"
    label: str                   # Human-readable label
    input_cost_per_mtok: float   # $ per 1M input tokens (fallback)
    output_cost_per_mtok: float  # $ per 1M output tokens (fallback)


# ── Default Model Configs ─────────────────────────────────────────────────────
# Which model handles which tier is a policy choice, so it lives here rather
# than being derived. Override per tier via settings.model.light/medium/heavy.
# Fallback prices verified against litellm's registry on 2026-08-04.

ANTHROPIC_TIERS = [
    ModelTier("claude-haiku-4-5", "light", "Haiku 4.5", 1.00, 5.00),
    # $2/$10 is introductory pricing through 2026-08-31; list is $3/$15.
    # The registry lookup tracks whichever is current.
    ModelTier("claude-sonnet-5", "medium", "Sonnet 5", 2.00, 10.00),
    # Opus is the cost-aware heavy default. Fable 5 ($10/$50) is opt-in via
    # --model / the extension picker — reserved for the hardest long-horizon work.
    ModelTier("claude-opus-5", "heavy", "Opus 5", 5.00, 25.00),
]

OPENAI_TIERS = [
    ModelTier("gpt-5.6-luna", "light", "GPT-5.6 Luna", 0.20, 1.20),
    ModelTier("gpt-5.6-terra", "medium", "GPT-5.6 Terra", 2.00, 12.00),
    ModelTier("gpt-5.6-sol", "heavy", "GPT-5.6 Sol", 5.00, 30.00),
]

GEMINI_TIERS = [
    ModelTier("gemini/gemini-3.5-flash-lite", "light", "Gemini 3.5 Flash-Lite", 0.30, 2.50),
    ModelTier("gemini/gemini-3.6-flash", "medium", "Gemini 3.6 Flash", 1.50, 7.50),
    ModelTier("gemini/gemini-3.1-pro-preview", "heavy", "Gemini 3.1 Pro", 2.00, 12.00),
]

# Provider configs keyed by prefix
PROVIDER_TIERS = {
    "anthropic": ANTHROPIC_TIERS,
    "openai": OPENAI_TIERS,
    "gemini": GEMINI_TIERS,
}


# ── Live pricing ──────────────────────────────────────────────────────────────
# litellm ships model_prices_and_context_window.json and refreshes it on every
# release, covering ~3,000 models. Reading prices from there means a new model
# or a price cut is picked up by `pip install -U litellm` — no edit here, and
# no AgentCode release. The static tiers above are only a fallback for when
# litellm is missing or doesn't list the model.

@lru_cache(maxsize=512)
def _registry_pricing(model: str) -> tuple[float, float] | None:
    """($ per 1M input, $ per 1M output) from litellm's registry, or None."""
    try:
        import litellm
    except ImportError:
        return None

    entry = litellm.model_cost.get(model)
    # Registry keys are sometimes unprefixed ("gemini-3.6-flash" for
    # "gemini/gemini-3.6-flash").
    if entry is None and "/" in model:
        entry = litellm.model_cost.get(model.split("/", 1)[1])
    if not entry:
        return None

    inp = entry.get("input_cost_per_token")
    out = entry.get("output_cost_per_token")
    if inp is None or out is None:
        return None
    return inp * 1_000_000, out * 1_000_000


def model_pricing(model: str) -> tuple[float, float] | None:
    """Price a model, preferring the live registry over the static tiers."""
    live = _registry_pricing(model)
    if live:
        return live
    for tiers in PROVIDER_TIERS.values():
        for tier in tiers:
            if tier.name == model:
                return tier.input_cost_per_mtok, tier.output_cost_per_mtok
    return None


def is_known_model(model: str) -> bool:
    """True if we can price the model — a cheap check for typos and retirements."""
    return model_pricing(model) is not None


# The registry lists the same model many times over — once per reseller and
# region (bedrock/…, snowflake/…, us.anthropic.…). Those are noise when you're
# browsing for a model to use, so hide them unless they're asked for by name.

# Dotted heads: region and Bedrock-style prefixes ("us.anthropic.claude-…").
_DOTTED_ALIAS_HEADS = {
    "us", "eu", "apac", "global", "au", "jp", "ca", "sa", "me", "il",
    "anthropic", "mistral", "meta", "cohere", "amazon", "ai21",
}

# Slashed heads: resellers and gateways ("snowflake/claude-…").
_SLASHED_ALIAS_HEADS = {
    "bedrock", "vertex_ai", "openrouter", "azure", "azure_ai", "replicate",
    "vercel_ai_gateway", "gradient_ai", "together_ai", "fireworks_ai",
    "sagemaker", "deepinfra", "hosted_vllm", "litellm_proxy", "nebius",
    "novita", "featherless_ai", "lambda_ai", "friendliai", "nscale",
    "snowflake", "databricks", "gmi", "cloudflare", "sambanova", "cerebras",
}

_ALIAS_HEADS = _DOTTED_ALIAS_HEADS | _SLASHED_ALIAS_HEADS


def _is_canonical(name: str) -> bool:
    """True for a first-party model id, False for a reseller/region alias."""
    if ":" in name:  # Bedrock ARN-style suffixes, e.g. ...-v1:0
        return False
    if "." in name and name.split(".", 1)[0] in _DOTTED_ALIAS_HEADS:
        return False
    if "/" in name and name.split("/", 1)[0] in _SLASHED_ALIAS_HEADS:
        return False
    return True


def list_registry_models(query: str = "", limit: int = 40) -> list[tuple[str, float, float]]:
    """Chat models litellm knows about, cheapest first, as (name, in, out).

    Reseller and regional aliases are hidden unless the query names one.
    """
    try:
        import litellm
    except ImportError:
        return []

    q = query.lower()
    include_aliases = bool(q) and any(
        q.startswith(p) or f"{p}/" in q or f"{p}." in q for p in _ALIAS_HEADS
    )

    rows: list[tuple[str, float, float]] = []
    for name, meta in litellm.model_cost.items():
        if meta.get("mode") != "chat":
            continue
        if q and q not in name.lower():
            continue
        if not include_aliases and not _is_canonical(name):
            continue
        inp = meta.get("input_cost_per_token")
        out = meta.get("output_cost_per_token")
        if not inp or not out:
            continue
        rows.append((name, inp * 1_000_000, out * 1_000_000))

    rows.sort(key=lambda r: r[1])
    return rows[:limit]


# ── Task Complexity Classification ────────────────────────────────────────────

# Patterns that signal increasing complexity
LIGHT_PATTERNS = [
    r"\b(explain|what is|what are|what does|how does|describe|list|show|tell me)\b",
    r"\b(read|cat|view|print|display|show me|open)\b",
    r"\b(git status|git log|git diff)\b",
    r"\b(hello|hi|hey|thanks|thank you)\b",
    r"\b(format|lint|indent|rename)\b",
    r"\b(typo|spelling|grammar|comment)\b",
]

MEDIUM_PATTERNS = [
    r"\b(write|create|add|implement|build|make|generate)\b",
    r"\b(fix|bug|error|debug|broken|failing|issue)\b",
    r"\b(test|unit test|pytest|spec)\b",
    r"\b(function|class|method|endpoint|route|api)\b",
    r"\b(edit|update|change|modify)\b",
    r"\b(install|setup|configure|deploy)\b",
]

HEAVY_PATTERNS = [
    r"\b(refactor|restructure|redesign|rearchitect|overhaul)\b",
    r"\b(migrate|migration|upgrade|convert|port)\b",
    r"\b(entire|whole|all files|every file|codebase|full)\b",
    r"\b(optimize|performance|bottleneck|profil)\b",
    r"\b(security|vulnerability|audit|review all)\b",
    r"\b(multi.?file|across files|multiple files)\b",
    r"\b(design pattern|architecture|system design)\b",
    r"\b(from scratch|ground up|complete|comprehensive)\b",
]


def classify_complexity(user_input: str) -> str:
    """
    Classify task complexity as 'light', 'medium', or 'heavy'.

    Uses pattern matching on the user's prompt. This is intentionally
    simple and transparent — users can see exactly why a model was chosen.
    """
    text = user_input.lower().strip()

    # Score each tier
    heavy_score = sum(1 for p in HEAVY_PATTERNS if re.search(p, text, re.IGNORECASE))
    medium_score = sum(1 for p in MEDIUM_PATTERNS if re.search(p, text, re.IGNORECASE))
    light_score = sum(1 for p in LIGHT_PATTERNS if re.search(p, text, re.IGNORECASE))

    # Length heuristic: very long prompts are usually complex
    if len(text) > 500:
        heavy_score += 2
    elif len(text) > 200:
        medium_score += 1

    # Multiple file references suggest complexity
    file_refs = len(re.findall(r'\.\w{1,4}\b', text))  # .py, .js, .tsx, etc.
    if file_refs >= 3:
        heavy_score += 2
    elif file_refs >= 2:
        medium_score += 1

    # Decision logic
    if heavy_score >= 2:
        return "heavy"
    elif heavy_score >= 1 and medium_score >= 1:
        return "heavy"
    elif medium_score >= 1:
        return "medium"
    else:
        return "light"


# ── Router ────────────────────────────────────────────────────────────────────

@dataclass
class CostTracker:
    """Track cumulative costs across a session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    requests: list[dict] = field(default_factory=list)
    last_turn_input: int = 0
    last_turn_output: int = 0
    last_turn_cost: float = 0.0

    def begin_turn(self):
        """Reset per-turn counters at the start of each agent loop call."""
        self.last_turn_input = 0
        self.last_turn_output = 0
        self.last_turn_cost = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.last_turn_input += input_tokens
        self.last_turn_output += output_tokens
        self.last_turn_cost += cost
        self.requests.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

    def summary(self) -> str:
        if not self.requests:
            return "No requests yet."
        lines = [
            f"Session total: ${self.total_cost:.4f} "
            f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out)",
            f"Requests: {len(self.requests)}",
        ]
        # Show last 5 requests
        for r in self.requests[-5:]:
            lines.append(
                f"  {r['model']}: ${r['cost']:.4f} "
                f"({r['input_tokens']:,} in / {r['output_tokens']:,} out)"
            )
        if len(self.requests) > 5:
            lines.append(f"  ... and {len(self.requests) - 5} earlier requests")
        return "\n".join(lines)


@dataclass
class ModelRouter:
    """
    Cost-aware model router.

    Automatically selects the cheapest model that can handle the task.
    Falls back to the default model if no tier config is available.
    """
    provider: str = "anthropic"       # "anthropic" or "openai"
    default_model: str = "claude-sonnet-5"
    enabled: bool = True              # False = always use default_model
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    # Allow per-tier overrides
    light_model: str | None = None
    medium_model: str | None = None
    heavy_model: str | None = None

    def get_tiers(self) -> list[ModelTier]:
        """Get the tier config for the current provider."""
        return PROVIDER_TIERS.get(self.provider, [])

    @staticmethod
    def detect_provider(model_string: str) -> str:
        """Detect provider from a model string."""
        m = model_string.lower()
        if "claude" in m or "anthropic" in m:
            return "anthropic"
        elif "gpt" in m or "openai" in m or "o1" in m or "o3" in m:
            return "openai"
        elif "gemini" in m or "google" in m:
            return "gemini"
        else:
            return "unknown"

    def route(self, user_input: str) -> tuple[str, str, str]:
        """
        Pick the best model for this task.

        Returns:
            (model_string, tier, reason) — the model to use, its tier,
            and a human-readable explanation of why it was chosen.
        """
        if not self.enabled:
            return self.default_model, "default", "routing disabled"

        complexity = classify_complexity(user_input)

        # Check for per-tier overrides first
        if complexity == "light" and self.light_model:
            return self.light_model, "light", self._reason(user_input, complexity)
        elif complexity == "medium" and self.medium_model:
            return self.medium_model, "medium", self._reason(user_input, complexity)
        elif complexity == "heavy" and self.heavy_model:
            return self.heavy_model, "heavy", self._reason(user_input, complexity)

        # Use provider tier defaults
        tiers = self.get_tiers()
        if not tiers:
            return self.default_model, "default", "no tier config for provider"

        tier_map = {t.tier: t for t in tiers}
        tier = tier_map.get(complexity)
        if tier:
            return tier.name, complexity, self._reason(user_input, complexity)

        return self.default_model, "default", "no matching tier"

    def _reason(self, user_input: str, complexity: str) -> str:
        """Generate a human-readable reason for the routing decision."""
        reasons = {
            "light": "simple query — using fast, cheap model",
            "medium": "standard coding task — using balanced model",
            "heavy": "complex multi-step task — using powerful model",
        }
        return reasons.get(complexity, "unknown complexity")

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request. Returns 0.0 for models we can't price."""
        pricing = model_pricing(model)
        if pricing is None:
            return 0.0
        input_cost, output_cost = pricing
        return (
            (input_tokens / 1_000_000) * input_cost
            + (output_tokens / 1_000_000) * output_cost
        )

    def get_tier_info(self, model: str) -> ModelTier | None:
        """Get tier info for a model."""
        for tiers in PROVIDER_TIERS.values():
            for tier in tiers:
                if tier.name == model:
                    return tier
        return None


def display_routing_decision(model: str, tier: str, reason: str, router: ModelRouter):
    """Show the user which model was selected and why."""
    tier_colors = {
        "light": "green",
        "medium": "yellow",
        "heavy": "red",
        "default": "cyan",
    }
    color = tier_colors.get(tier, "white")

    pricing = model_pricing(model)
    cost_hint = ""
    if pricing:
        cost_hint = f" [dim](${pricing[0]:.2f}/${pricing[1]:.2f} per 1M tok)[/dim]"

    console.print(
        f"  [{color}]⚡ {tier.upper()}[/{color}] → "
        f"[bold]{model}[/bold]{cost_hint} "
        f"[dim]({reason})[/dim]"
    )