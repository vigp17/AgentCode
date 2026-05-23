"""
AgentCode - Cost-aware model router.

Automatically picks the cheapest model that can handle the task.
Simple questions get a fast, cheap model. Complex multi-file refactors
get a powerful, expensive one.
"""

import re
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


# ── Model Tiers ───────────────────────────────────────────────────────────────

@dataclass
class ModelTier:
    """A model with its capabilities and pricing."""
    name: str                    # LiteLLM model string
    tier: str                    # "light", "medium", "heavy"
    label: str                   # Human-readable label
    input_cost_per_mtok: float   # $ per 1M input tokens
    output_cost_per_mtok: float  # $ per 1M output tokens


# ── Default Model Configs ─────────────────────────────────────────────────────
# Users can override these via AGENTCODE.md or environment variables.
# Costs are approximate as of mid-2026.

ANTHROPIC_TIERS = [
    ModelTier("claude-haiku-4-5-20251001", "light", "Haiku 4.5", 0.80, 4.00),
    ModelTier("claude-sonnet-4-6", "medium", "Sonnet 4.6", 3.00, 15.00),
    ModelTier("claude-opus-4-6", "heavy", "Opus 4.6", 15.00, 75.00),
]

OPENAI_TIERS = [
    ModelTier("gpt-4o-mini", "light", "GPT-4o Mini", 0.15, 0.60),
    ModelTier("gpt-4o", "medium", "GPT-4o", 2.50, 10.00),
    ModelTier("gpt-5.5", "heavy", "GPT-5.5", 15.00, 60.00),
]

GEMINI_TIERS = [
    ModelTier("gemini/gemini-2.0-flash", "light", "Gemini 2.0 Flash", 0.10, 0.40),
    ModelTier("gemini/gemini-2.5-flash", "medium", "Gemini 2.5 Flash", 0.25, 1.00),
    ModelTier("gemini/gemini-2.5-pro", "heavy", "Gemini 2.5 Pro", 1.25, 10.00),
]

# Azure deployment names are user-defined, so tiers use common defaults.
# Users can override via settings.json model.light/medium/heavy.
AZURE_TIERS = [
    ModelTier("azure/gpt-4o-mini", "light", "Azure GPT-4o Mini", 0.15, 0.60),
    ModelTier("azure/gpt-4o", "medium", "Azure GPT-4o", 2.50, 10.00),
    ModelTier("azure/gpt-4o", "heavy", "Azure GPT-4o", 2.50, 10.00),
]

# Provider configs keyed by prefix
PROVIDER_TIERS = {
    "anthropic": ANTHROPIC_TIERS,
    "openai": OPENAI_TIERS,
    "gemini": GEMINI_TIERS,
    "azure": AZURE_TIERS,
}


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
    default_model: str = "claude-sonnet-4-6"
    enabled: bool = True              # False = always use default_model
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    # Allow per-tier overrides
    light_model: str | None = None
    medium_model: str | None = None
    heavy_model: str | None = None

    def get_tiers(self) -> list[ModelTier]:
        """Get the tier config for the current provider."""
        return PROVIDER_TIERS.get(self.provider, [])

    def detect_provider(self, model_string: str) -> str:
        """Detect provider from a model string."""
        m = model_string.lower()
        if m.startswith("azure/"):
            return "azure"
        elif "claude" in m or "anthropic" in m:
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
        """Estimate cost for a request."""
        # Search all tiers for this model
        for tiers in PROVIDER_TIERS.values():
            for tier in tiers:
                if tier.name == model:
                    cost = (
                        (input_tokens / 1_000_000) * tier.input_cost_per_mtok
                        + (output_tokens / 1_000_000) * tier.output_cost_per_mtok
                    )
                    return cost
        return 0.0  # Unknown model, can't estimate

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

    tier_info = router.get_tier_info(model)
    cost_hint = ""
    if tier_info:
        cost_hint = f" [dim](${tier_info.input_cost_per_mtok:.2f}/${tier_info.output_cost_per_mtok:.2f} per 1M tok)[/dim]"

    console.print(
        f"  [{color}]⚡ {tier.upper()}[/{color}] → "
        f"[bold]{model}[/bold]{cost_hint} "
        f"[dim]({reason})[/dim]"
    )