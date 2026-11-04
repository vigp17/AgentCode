"""Tests for AgentCode's fine-tuned model registry, trajectory logging and data prep."""

import json
import sys
from pathlib import Path

import pytest

import finetuned
import trajectories
from router import ModelRouter, model_pricing
from settings import Settings, TrainingSettings, _dict_to_settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
import prepare_data  # noqa: E402


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    for k in ("AGENTCODE_FT_API_BASE", "AGENTCODE_FT_LIGHT_API_BASE",
              "AGENTCODE_FT_MEDIUM_API_BASE", "AGENTCODE_FT_HEAVY_API_BASE"):
        monkeypatch.delenv(k, raising=False)
    finetuned.configure({})
    yield
    finetuned.configure({})


def test_three_builtin_finetunes_one_per_tier():
    models = [m for m in finetuned.all_models(include_unconfigured=True)
              if m.name.startswith("agentcode-")]
    assert sorted(m.name for m in models) == ["agentcode-heavy", "agentcode-light", "agentcode-medium"]
    assert {m.tier for m in models} == {"light", "medium", "heavy"}
    assert all(m.base_model.startswith("Qwen/") for m in models)


def test_unconfigured_finetunes_are_hidden_from_picker():
    assert finetuned.all_models() == []
    # Only the ecosystem lineup is always listed — marked unavailable.
    rows = finetuned.as_discovered()
    assert [r.id for r in rows] == ["axon-1", "tandav-1", "veda-1", "chitra-1"]
    assert not any(r.available for r in rows)


def test_env_configures_endpoints(monkeypatch):
    monkeypatch.setenv("AGENTCODE_FT_API_BASE", "http://gpu:9000/v1")
    monkeypatch.setenv("AGENTCODE_FT_HEAVY_API_BASE", "http://big-gpu:9000/v1")
    finetuned.configure({})
    assert finetuned.get("agentcode-light").api_base == "http://gpu:9000/v1"
    assert finetuned.get("agentcode-heavy").api_base == "http://big-gpu:9000/v1"
    assert len(finetuned.all_models()) == 3


def test_completion_kwargs_route_to_endpoint(monkeypatch):
    finetuned.configure({"agentcode-medium": {"api_base": "http://h:8002/v1", "served_name": "ac-med"}})
    kw = finetuned.completion_kwargs("agentcode-medium")
    assert kw["model"] == "openai/ac-med"
    assert kw["api_base"] == "http://h:8002/v1"
    assert kw["api_key"]  # litellm's openai client needs a non-empty key
    assert finetuned.completion_kwargs("claude-sonnet-5") == {"model": "claude-sonnet-5"}


def test_settings_can_add_extra_finetunes():
    finetuned.configure({"my-sql-expert": {"tier": "medium", "api_base": "http://x/v1",
                                           "input_cost_per_mtok": 0.5, "output_cost_per_mtok": 1.5}})
    assert finetuned.is_finetuned("my-sql-expert")
    assert model_pricing("my-sql-expert") == (0.5, 1.5)


def test_router_routes_between_finetunes():
    finetuned.configure({n: {"api_base": "http://h/v1"} for n in
                         ("agentcode-light", "agentcode-medium", "agentcode-heavy")})
    r = ModelRouter(provider=ModelRouter.detect_provider("agentcode-medium"),
                    default_model="agentcode-medium")
    assert r.provider == "agentcode"
    assert r.route("explain this")[0] == "agentcode-light"
    assert r.route("refactor the entire codebase")[0] == "agentcode-heavy"
    assert model_pricing("agentcode-heavy") == (0.0, 0.0)


def test_settings_parse_finetuned_and_training():
    s = _dict_to_settings({
        "finetuned": {"agentcode-light": {"api_base": "http://x/v1"}},
        "training": {"log_trajectories": True},
        "model": {"auto_tiers": False},
    })
    assert s.finetuned["agentcode-light"]["api_base"] == "http://x/v1"
    assert s.training.log_trajectories is True
    assert s.model.auto_discover is True and s.model.auto_tiers is False


# ── trajectories ──────────────────────────────────────────────────────────────

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}},
         {"type": "function", "function": {"name": "mcp__gh__x", "parameters": {}}}]


def _turn():
    user = {"role": "user", "content": "what's in a.py?"}
    return user, [
        {"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"},
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "print(1)"},
        {"role": "assistant", "content": "It prints 1."},
    ]


def test_trajectory_logging_off_by_default(tmp_path):
    user, msgs = _turn()
    s = Settings(training=TrainingSettings(trajectories_dir=str(tmp_path)))
    assert trajectories.log_turn(s, "sys", msgs, TOOLS, "m", "/p") is None
    assert not list(tmp_path.iterdir())


def test_trajectory_logs_only_this_turn(tmp_path):
    user, msgs = _turn()
    s = Settings(training=TrainingSettings(log_trajectories=True, trajectories_dir=str(tmp_path)))
    path = trajectories.log_turn(s, "sys", trajectories.turn_slice(msgs, user), TOOLS, "m", "/proj")
    rec = json.loads(path.read_text())
    assert rec["messages"][0] == {"role": "system", "content": "sys"}
    assert rec["messages"][1]["content"] == "what's in a.py?"
    assert len(rec["messages"]) == 5
    assert rec["meta"]["tool_calls"] == 1
    assert "/proj" not in path.read_text()  # project is hashed


def test_prepare_data_cleans_and_strips_mcp_tools():
    user, msgs = _turn()
    rec = {"messages": [{"role": "system", "content": "s"}, *trajectories.turn_slice(msgs, user)],
           "tools": TOOLS, "meta": {}}
    out = prepare_data.clean(rec)
    assert out and [t["function"]["name"] for t in out["tools"]] == ["read_file"]


@pytest.mark.parametrize("mutate", [
    lambda m: m[2]["tool_calls"][0]["function"].update(name="rm_rf"),        # unknown tool
    lambda m: m[2]["tool_calls"][0]["function"].update(arguments="{bad"),     # bad JSON
    lambda m: m.pop(3),                                                       # missing result
    lambda m: m.pop(),                                                        # no final answer
])
def test_prepare_data_rejects_broken_trajectories(mutate):
    user, msgs = _turn()
    turn = [{"role": "system", "content": "s"}, *json.loads(json.dumps(trajectories.turn_slice(msgs, user)))]
    mutate(turn)
    assert prepare_data.clean({"messages": turn, "tools": TOOLS}) is None


def test_tier_slices():
    user, msgs = _turn()
    rec = prepare_data.clean({"messages": [{"role": "system", "content": "s"},
                                           *trajectories.turn_slice(msgs, user)], "tools": TOOLS})
    assert prepare_data.fits_tier(rec, {"max_tool_calls": 3, "max_chars": 24000})
    assert not prepare_data.fits_tier(rec, {"max_tool_calls": 0, "max_chars": None})
    for tier in ("light", "medium", "heavy"):
        assert prepare_data.load_config(tier)["name"] == f"agentcode-{tier}"


# ── model sizes & training loss mask ──────────────────────────────────────────

import train  # noqa: E402  (finetune/ is on sys.path above; no GPU deps at import)


def test_bases_match_requested_sizes():
    bases = {m.tier: m.base_model for m in finetuned.all_models(include_unconfigured=True)
             if m.name.startswith("agentcode-")}
    assert bases == {
        "light": "Qwen/Qwen3-4B-Instruct-2507",            # 4B
        "medium": "Qwen/Qwen3-14B",                        # ~12B class
        "heavy": "Qwen/Qwen3-Next-80B-A3B-Instruct",       # 80B MoE
    }
    for tier, base in bases.items():  # registry and training configs agree
        assert prepare_data.load_config(tier)["base_model"] == base


def test_assistant_spans_cover_only_replies():
    text = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>{}</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>R</tool_response><|im_end|>\n"
            "<|im_start|>assistant\nDone.<|im_end|>\n")
    spans = [text[a:b] for a, b in train.assistant_spans(text)]
    assert spans == ["<tool_call>{}</tool_call><|im_end|>", "Done.<|im_end|>"]


def test_mask_labels_ignores_context_tokens():
    text = "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\nyo<|im_end|>"
    # Fake one-token-per-char tokenizer.
    ids = list(range(len(text)))
    offsets = [(i, i + 1) for i in range(len(text))]
    labels = train.mask_labels(ids, offsets, train.assistant_spans(text))
    kept = "".join(text[i] for i, lab in enumerate(labels) if lab != -100)
    assert kept == "yo<|im_end|>"


# ── ecosystem: axon-1 / tandav-1 / veda-1 / chitra-1 ──────────────────────────

def test_ecosystem_metadata():
    axon, tandav, veda, chitra = (finetuned.get(n) for n in ("axon-1", "tandav-1", "veda-1", "chitra-1"))
    assert axon.supports_effort and not tandav.supports_effort
    assert tandav.max_iterations == 50
    assert chitra.kind == "image" and veda.kind == "chat"


def test_ecosystem_endpoint_env(monkeypatch):
    monkeypatch.setenv("AGENTCODE_FT_AXON_1_API_BASE", "http://localhost:11434/v1")
    finetuned.configure({})
    rows = {r.id: r for r in finetuned.as_discovered()}
    assert rows["axon-1"].available and not rows["tandav-1"].available
    assert finetuned.completion_kwargs("axon-1")["api_base"] == "http://localhost:11434/v1"
    assert ModelRouter.detect_provider("axon-1") == "agentcode"
