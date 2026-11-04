"""
AgentCode — automatic model discovery.

Asks Anthropic, OpenAI and Google which models your API keys can use, so a
newly released model shows up in `/models`, the VS Code picker and (optionally)
the routing tiers without anyone editing AgentCode.

Sources, in order of preference for each provider:
  1. The provider's own `/models` endpoint (needs that provider's API key).
     This is the only source that knows about a model the day it ships, and it
     also lists your own fine-tunes (OpenAI `ft:` models).
  2. litellm's model registry, filtered to that provider's first-party ids.
     Used when there is no key or the request fails.

Results are cached in ~/.agentcode/models_cache.json (default TTL 24h, set
AGENTCODE_MODELS_TTL_HOURS) so startup never waits on the network more than
once a day. Set AGENTCODE_DISABLE_DISCOVERY=1 to turn it off entirely.

Only the standard library is used for HTTP, so this adds no dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

PROVIDERS = ("anthropic", "openai", "gemini")

PROVIDER_LABELS = {
    "anthropic": "Claude",
    "openai": "GPT",
    "gemini": "Gemini",
    "agentcode": "AgentCode (fine-tuned)",
}

API_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

_HTTP_TIMEOUT = 8  # seconds per request


@dataclass
class DiscoveredModel:
    id: str                    # model string AgentCode passes to litellm
    provider: str              # "anthropic" | "openai" | "gemini" | "agentcode"
    label: str                 # human-readable name
    created: float | None = None   # unix time, when the provider reports it
    source: str = "api"        # "api" | "registry" | "finetuned"


# ── Config helpers ────────────────────────────────────────────────────────────

def discovery_disabled() -> bool:
    return os.environ.get("AGENTCODE_DISABLE_DISCOVERY", "").lower() in ("1", "true", "yes")


def _api_key(provider: str) -> str | None:
    for var in API_KEY_ENV.get(provider, ()):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _key_fingerprint(key: str | None) -> str:
    """Cache entries are tied to the key that fetched them (different accounts
    can see different models) without ever writing the key itself to disk."""
    if not key:
        return "nokey"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _cache_path() -> Path:
    override = os.environ.get("AGENTCODE_MODELS_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".agentcode" / "models_cache.json"


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get("AGENTCODE_MODELS_TTL_HOURS", "24")) * 3600
    except ValueError:
        return 24 * 3600


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "agentcode", **headers})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Provider fetchers ─────────────────────────────────────────────────────────
# Each returns every *chat-capable* model the key can use, newest first where
# the provider tells us creation time.

def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def fetch_anthropic(key: str) -> list[DiscoveredModel]:
    models: list[DiscoveredModel] = []
    after: str | None = None
    for _ in range(20):  # pagination guard
        url = "https://api.anthropic.com/v1/models?limit=1000"
        if after:
            url += f"&after_id={after}"
        data = _get_json(url, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            models.append(DiscoveredModel(
                id=mid,
                provider="anthropic",
                label=m.get("display_name") or mid,
                created=_parse_iso(m.get("created_at")),
            ))
        if not data.get("has_more") or not data.get("last_id"):
            break
        after = data["last_id"]
    return models


# OpenAI's /models lists everything the key can call — embeddings, audio,
# image, moderation... Keep only what works with chat completions + tools.
_OPENAI_CHAT_PREFIX = re.compile(r"^(gpt-|o\d|chatgpt-|ft:gpt-|ft:o\d)")
_OPENAI_EXCLUDE = (
    "embedding", "tts", "whisper", "dall-e", "image", "audio", "realtime",
    "transcribe", "moderation", "search", "instruct", "davinci", "babbage",
    "codex", "computer-use", "deep-research",
)
_DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def is_openai_chat_model(mid: str) -> bool:
    m = mid.lower()
    if not _OPENAI_CHAT_PREFIX.match(m):
        return False
    return not any(x in m for x in _OPENAI_EXCLUDE)


def fetch_openai(key: str) -> list[DiscoveredModel]:
    data = _get_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
    ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
    created = {m["id"]: m.get("created") for m in data.get("data", []) if m.get("id")}
    models = []
    for mid in ids:
        if not is_openai_chat_model(mid):
            continue
        # Hide dated snapshots ("gpt-5.6-terra-2026-07-01") when the moving
        # alias is also listed — the alias is what people want to pick.
        if _DATED_SNAPSHOT.search(mid) and _DATED_SNAPSHOT.sub("", mid) in ids:
            continue
        label = mid
        if mid.startswith("ft:"):
            # ft:gpt-…:org:suffix:id → "Fine-tune: suffix (gpt-…)"
            parts = mid.split(":")
            suffix = parts[3] if len(parts) > 3 and parts[3] else parts[-1]
            label = f"Fine-tune: {suffix} ({parts[1]})"
        models.append(DiscoveredModel(
            id=mid, provider="openai", label=label, created=created.get(mid),
        ))
    return models


_GEMINI_EXCLUDE = (
    "embedding", "aqa", "imagen", "veo", "tts", "image", "live",
    "native-audio", "robotics", "computer-use", "learnlm", "deep-research",
)
_GEMINI_EXP_SNAPSHOT = re.compile(r"-exp-\d{4}$")  # gemini-exp-1206: old experiments


def is_gemini_chat_model(name: str, methods: list[str]) -> bool:
    n = name.lower().removeprefix("gemini/")
    if not n.startswith("gemini") or "generateContent" not in methods:
        return False
    if _GEMINI_EXP_SNAPSHOT.search(n):
        return False
    return not any(x in n for x in _GEMINI_EXCLUDE)


def fetch_gemini(key: str) -> list[DiscoveredModel]:
    models: list[DiscoveredModel] = []
    token: str | None = None
    for _ in range(20):
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        if token:
            url += f"&pageToken={token}"
        data = _get_json(url, {"x-goog-api-key": key})
        for m in data.get("models", []):
            name = (m.get("name") or "").removeprefix("models/")
            if not name or not is_gemini_chat_model(name, m.get("supportedGenerationMethods", [])):
                continue
            models.append(DiscoveredModel(
                id=f"gemini/{name}", provider="gemini",
                label=m.get("displayName") or name,
            ))
        token = data.get("nextPageToken")
        if not token:
            break
    return models


FETCHERS = {
    "anthropic": fetch_anthropic,
    "openai": fetch_openai,
    "gemini": fetch_gemini,
}


# ── Registry fallback ─────────────────────────────────────────────────────────

def registry_models(provider: str) -> list[DiscoveredModel]:
    """First-party chat models litellm knows for a provider (no key needed)."""
    try:
        import litellm
    except ImportError:
        return []
    out = []
    for name, meta in litellm.model_cost.items():
        if meta.get("mode") != "chat":
            continue
        if meta.get("litellm_provider") != provider:
            continue
        if provider == "gemini" and not (
            name.startswith("gemini/") and is_gemini_chat_model(name, ["generateContent"])
        ):
            continue
        if provider == "openai" and not is_openai_chat_model(name):
            continue
        if ":" in name or "/" in name.removeprefix("gemini/"):
            continue
        out.append(DiscoveredModel(id=name, provider=provider, label=name, source="registry"))
    return out


# ── Cache ─────────────────────────────────────────────────────────────────────

_cache_lock = threading.Lock()


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass  # read-only home etc. — discovery still works, just uncached


def _from_cache_entry(entry: dict) -> list[DiscoveredModel]:
    return [DiscoveredModel(**m) for m in entry.get("models", [])]


def discover_provider(provider: str, refresh: bool = False) -> list[DiscoveredModel]:
    """Models for one provider: cached API result, fresh API result, or registry."""
    key = _api_key(provider)
    if not key:
        return registry_models(provider)

    fp = _key_fingerprint(key)
    with _cache_lock:
        cache = _load_cache()
    entry = cache.get(provider, {})
    fresh = (
        entry.get("key") == fp
        and time.time() - entry.get("fetched_at", 0) < _ttl_seconds()
    )
    if fresh and not refresh:
        return _from_cache_entry(entry)

    try:
        models = FETCHERS[provider](key)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        # Offline / bad key / API hiccup: stale cache beats nothing, and the
        # registry beats an empty picker.
        if entry.get("key") == fp and entry.get("models"):
            return _from_cache_entry(entry)
        return registry_models(provider)

    with _cache_lock:
        cache = _load_cache()
        cache[provider] = {
            "key": fp,
            "fetched_at": time.time(),
            "models": [asdict(m) for m in models],
        }
        _save_cache(cache)
    return models


def discover_all(refresh: bool = False) -> dict[str, list[DiscoveredModel]]:
    """Discover every provider in parallel. Never raises."""
    if discovery_disabled():
        return {p: [] for p in PROVIDERS}
    results: dict[str, list[DiscoveredModel]] = {}

    def run(p: str) -> None:
        try:
            results[p] = discover_provider(p, refresh=refresh)
        except Exception:
            results[p] = []

    threads = [threading.Thread(target=run, args=(p,), daemon=True) for p in PROVIDERS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_HTTP_TIMEOUT * 3)
    return {p: _sort_newest(results.get(p, [])) for p in PROVIDERS}


# ── Ordering & auto-tiering ───────────────────────────────────────────────────

def _version(mid: str) -> tuple[float, ...]:
    """Numeric version found in a model id: 'claude-opus-5-5' → (5, 5),
    'gemini-3.6-flash' → (3, 6), 'gpt-5.6-sol' → (5, 6). Dates are ignored."""
    core = _DATED_SNAPSHOT.sub("", mid.split("/")[-1])
    core = re.sub(r"-\d{8}$", "", core)  # Anthropic-style 20250929 suffix
    nums = re.findall(r"(?<![a-z])(\d+(?:[.-]\d+)*)", core)
    if not nums:
        return ()
    return tuple(float(x) for x in re.split(r"[.-]", nums[0]) if x)


def _is_preview(mid: str) -> bool:
    return any(t in mid for t in ("preview", "exp", "beta", "latest"))


def _sort_newest(models: list[DiscoveredModel]) -> list[DiscoveredModel]:
    return sorted(
        models,
        key=lambda m: (m.created or 0, _version(m.id), not _is_preview(m.id), m.id),
        reverse=True,
    )


def _newest(models: list[DiscoveredModel]) -> DiscoveredModel | None:
    return _sort_newest(models)[0] if models else None


def _anthropic_tiers(models: list[DiscoveredModel]) -> dict[str, str]:
    families = {"light": "haiku", "medium": "sonnet", "heavy": "opus"}
    out = {}
    for tier, fam in families.items():
        best = _newest([m for m in models if fam in m.id])
        if best:
            out[tier] = best.id
    return out


def _gemini_tiers(models: list[DiscoveredModel]) -> dict[str, str]:
    def pick(pred) -> str | None:
        cands = [m for m in models if pred(m.id)]
        # No creation time from Google — newest version wins, stable beats
        # preview at the same version.
        # Shorter id wins a tie: "…-pro-preview" over "…-pro-preview-customtools".
        cands.sort(key=lambda m: (_version(m.id), not _is_preview(m.id), -len(m.id)), reverse=True)
        return cands[0].id if cands else None

    out = {
        "light": pick(lambda i: "flash-lite" in i),
        "medium": pick(lambda i: "flash" in i and "lite" not in i),
        "heavy": pick(lambda i: "-pro" in i),
    }
    return {k: v for k, v in out.items() if v}


_OPENAI_TIER_WORDS = {
    "light": ("luna", "nano", "mini"),
    "medium": ("terra",),
    "heavy": ("sol",),
}


def _openai_tiers(models: list[DiscoveredModel], price) -> dict[str, str]:
    """Newest model of each OpenAI size class (luna/terra/sol, nano/mini).

    If a release uses names we don't recognize, fall back to ranking the newest
    gpt-X.Y family by price: cheapest → light, priciest → heavy.
    """
    skip = ("-pro", "chat-latest", "oss", "astra", "rosalind")
    cands = [
        m for m in models
        if m.id.startswith("gpt-") and not _DATED_SNAPSHOT.search(m.id)
        and not any(s in m.id for s in skip)
    ]
    if not cands:
        return {}

    def newest(words) -> str | None:
        hits = [m for m in cands if any(re.search(rf"-{w}\b", m.id) for w in words)]
        hits.sort(key=lambda m: (_version(m.id), m.created or 0, -len(m.id)), reverse=True)
        return hits[0].id if hits else None

    out = {tier: newest(words) for tier, words in _OPENAI_TIER_WORDS.items()}
    out = {k: v for k, v in out.items() if v}
    if len(out) == 3:
        return out

    newest_ver = max(_version(m.id)[:2] for m in cands)
    family = [m.id for m in cands if _version(m.id)[:2] == newest_ver]
    priced = sorted((p, mid) for mid in family if (p := price(mid)))
    if len(priced) < 3:
        return out
    return {"light": priced[0][1], "medium": priced[len(priced) // 2][1], "heavy": priced[-1][1]}


def auto_tiers(provider: str, models: list[DiscoveredModel], price=None) -> dict[str, str]:
    """Pick the newest light/medium/heavy model for a provider from discovered
    models. Returns only the tiers it is confident about."""
    if provider == "anthropic":
        return _anthropic_tiers(models)
    if provider == "gemini":
        return _gemini_tiers(models)
    if provider == "openai":
        if price is None:
            from router import model_pricing as price
        return _openai_tiers(models, price)
    return {}


# ── Background refresh ────────────────────────────────────────────────────────

def refresh_in_background(callback=None, refresh: bool = False) -> threading.Thread:
    """Discover in a daemon thread; call callback(results) when done."""
    def run():
        results = discover_all(refresh=refresh)
        if callback:
            try:
                callback(results)
            except Exception:
                pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def to_wire(results: dict[str, list[DiscoveredModel]], extra: list[DiscoveredModel] = ()) -> list[dict]:
    """Flatten for the VS Code protocol, fine-tunes first."""
    rows = [asdict(m) for m in extra]
    for p in PROVIDERS:
        rows.extend(asdict(m) for m in results.get(p, []))
    return rows
