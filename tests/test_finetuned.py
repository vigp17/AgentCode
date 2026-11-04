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
    for k in ("AGENTCODE_FT_AXON_1_API_BASE", "AGENTCODE_FT_TANDAV_1_API_BASE",
              "AGENTCODE_FT_VEDA_1_API_BASE", "AGENTCODE_FT_CHITRA_1_API_BASE"):
        monkeypatch.delenv(k, raising=False)
    finetuned.configure({})
    yield
    finetuned.configure({})


def test_lineup_is_the_only_builtin_set():
    models = finetuned.all_models(include_unconfigured=True)
    assert [m.name for m in models] == ["axon-1", "tandav-1", "veda-1", "chitra-1"]
    assert {m.name: m.tier for m in models}["axon-1"] == "medium"
    assert {m.name: m.tier for m in models}["tandav-1"] == "heavy"


def test_unconfigured_finetunes_are_hidden_from_picker():
    assert finetuned.all_models() == []
    # Only the ecosystem lineup is always listed — marked unavailable.
    rows = finetuned.as_discovered()
    assert [r.id for r in rows] == ["axon-1", "tandav-1", "veda-1", "chitra-1"]
    assert not any(r.available for r in rows)


def test_env_configures_endpoints(monkeypatch):
    monkeypatch.setenv("AGENTCODE_FT_TANDAV_1_API_BASE", "http://big-gpu:9000/v1")
    finetuned.configure({})
    assert finetuned.get("tandav-1").api_base == "http://big-gpu:9000/v1"
    assert [m.name for m in finetuned.all_models()] == ["tandav-1"]


def test_completion_kwargs_route_to_endpoint(monkeypatch):
    finetuned.configure({"axon-1": {"api_base": "http://h:8002/v1", "served_name": "qwen2.5-coder:14b"}})
    kw = finetuned.completion_kwargs("axon-1")
    assert kw["model"] == "openai/qwen2.5-coder:14b"
    assert kw["api_base"] == "http://h:8002/v1"
    assert kw["api_key"]  # litellm's openai client needs a non-empty key
    assert finetuned.completion_kwargs("claude-sonnet-5") == {"model": "claude-sonnet-5"}


def test_settings_can_add_extra_finetunes():
    finetuned.configure({"my-sql-expert": {"tier": "medium", "api_base": "http://x/v1",
                                           "input_cost_per_mtok": 0.5, "output_cost_per_mtok": 1.5}})
    assert finetuned.is_finetuned("my-sql-expert")
    assert model_pricing("my-sql-expert") == (0.5, 1.5)


def test_router_routes_between_lineup_models():
    finetuned.configure({n: {"api_base": "http://h/v1"} for n in ("axon-1", "tandav-1", "veda-1")})
    r = ModelRouter(provider=ModelRouter.detect_provider("axon-1"), default_model="axon-1")
    assert r.provider == "agentcode"
    assert r.route("write a function to parse dates")[0] == "axon-1"
    assert r.route("refactor the entire codebase")[0] == "tandav-1"
    assert r.route("explain this")[0] == "axon-1"  # no light model → default
    assert model_pricing("tandav-1") == (0.0, 0.0)


def test_settings_parse_finetuned_and_training():
    s = _dict_to_settings({
        "finetuned": {"axon-1": {"api_base": "http://x/v1"}},
        "training": {"log_trajectories": True},
        "model": {"auto_tiers": False},
    })
    assert s.finetuned["axon-1"]["api_base"] == "http://x/v1"
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


def _rec(n_calls=1, answer="It prints 1.", tool="read_file", tool_result="print(1)", verify=False):
    """A clean trajectory with n_calls tool calls."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n_calls):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": tool, "arguments": json.dumps({"path": f"f{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": tool_result})
    if verify:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "v", "type": "function",
             "function": {"name": "run_command", "arguments": '{"command": "pytest -q"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": "v", "content": "3 passed"})
    msgs.append({"role": "assistant", "content": answer})
    tools = [{"type": "function", "function": {"name": n, "parameters": {}}}
             for n in ("read_file", "edit_file", "run_command", "search_text")]
    return prepare_data.clean({"messages": msgs, "tools": tools, "meta": {"project": "p"}})


def test_effort_labels_follow_work_done():
    assert prepare_data.effort_for(_rec(1)) == "low"
    assert prepare_data.effort_for(_rec(5)) == "medium"
    assert prepare_data.effort_for(_rec(12)) == "high"


def test_axon_examples_carry_the_same_effort_tag_the_server_sends():
    rec = prepare_data.with_effort_tag(_rec(5))
    assert rec["messages"][0]["content"] == "sys" + finetuned.effort_system_suffix("medium")
    assert rec["meta"]["effort"] == "medium"


def test_veda_slice_keeps_only_read_only_research():
    long_answer = "Spec: " + "x" * 900
    research = _rec(3, answer=long_answer, tool="search_text")
    coding = _rec(3, answer=long_answer, tool="edit_file")
    short = _rec(3, answer="ok", tool="read_file")
    veda = prepare_data.load_config("veda-1")
    assert prepare_data.build_slice([research, coding, short], veda) == [research]


def test_slices_per_model():
    axon, tandav = prepare_data.load_config("axon-1"), prepare_data.load_config("tandav-1")
    tiny, big = _rec(1), _rec(60)
    assert [r["meta"].get("effort") for r in prepare_data.build_slice([tiny, big], axon)] == ["low"]
    assert prepare_data.build_slice([tiny, big], tandav) == [big]  # min 3 tool calls, no max


def test_reward_prefers_verified_clean_work():
    edited_then_tested = _rec(2, tool="edit_file", verify=True,
                              answer="Fixed the off-by-one in the date parser; added a regression test and the suite passes.")
    errors = _rec(3, tool="read_file", tool_result="Error: File not found: x.py")
    assert prepare_data.reward(edited_then_tested) == 0.8
    assert prepare_data.reward(errors) < 0.5


def test_reward_penalizes_loops():
    rec = _rec(1)
    call = rec["messages"][2]
    for i in range(3):  # same call repeated
        rec["messages"].insert(-1, {**call, "tool_calls": [{**call["tool_calls"][0], "id": f"r{i}"}]})
        rec["messages"].insert(-1, {"role": "tool", "tool_call_id": f"r{i}", "content": "print(1)"})
    assert prepare_data.reward(rec) <= 0.4


# ── model sizes & training loss mask ──────────────────────────────────────────

import train  # noqa: E402  (finetune/ is on sys.path above; no GPU deps at import)


def test_registry_and_training_configs_agree():
    for name in ("axon-1", "tandav-1", "veda-1"):
        assert prepare_data.load_config(name)["base_model"] == finetuned.get(name).base_model
    assert prepare_data.load_config("tandav-1")["lora"]["r"] == 128  # plan: high-rank adapters


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
