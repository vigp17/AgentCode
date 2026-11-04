#!/usr/bin/env python3
"""
Turn AgentCode trajectories into train/eval sets for the lineup.

    python finetune/prepare_data.py                         # axon-1, tandav-1, veda-1
    python finetune/prepare_data.py --model axon-1
    python finetune/prepare_data.py --input finetune/raw --min-reward 0.6 --top 500

Input: JSONL files written by AgentCode when settings.training.log_trajectories
is on (default dir ~/.agentcode/trajectories), or by generate_data.py. Each line
is {"messages": [...], "tools": [...], "meta": {...}} in OpenAI chat format.

What it does:
  * drops turns whose tool calls are malformed (unknown tool, non-JSON args)
    or whose tool_call ids don't line up with tool results
  * scores every trajectory with a heuristic reward (see reward()) and keeps
    only those above --min-reward, best first, optionally capped by --top
  * truncates oversized tool results and de-duplicates
  * builds a slice per model:
      axon-1    everyday trajectories, each labelled with an <effort> tag
                (low/medium/high, from how much work it took) — the same tag
                server.py adds when you pick an effort in VS Code
      tandav-1  everything with real work in it, including the longest
      veda-1    research trajectories: read/search-only tool use ending in a
                substantial written answer
  * splits train/eval *by project* so eval measures generalization, not recall

Output: finetune/data/<model>/{train,eval}.jsonl  (stdlib only — runs anywhere)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = Path.home() / ".agentcode" / "trajectories"
MAX_TOOL_RESULT_CHARS = 6000
MODELS = ("axon-1", "tandav-1", "veda-1")

# Must match finetuned.effort_system_suffix() — server.py appends exactly this
# to axon-1's system prompt at inference (a test keeps them in sync).
EFFORT_SUFFIX = "\n\n<effort>{effort}</effort>"
# Effort a trajectory "needed", by tool calls. Aligned with the server's
# iteration caps for each level (8 / 25 / 40).
EFFORT_BY_TOOL_CALLS = ((2, "low"), (8, "medium"))  # else "high"

READ_ONLY_TOOLS = {
    "read_file", "list_directory", "search_files", "search_text",
    "git_status", "git_log", "git_diff", "web_search", "fetch_url",
}
VERIFY_RE = re.compile(
    r"\b(pytest|unittest|npm (run )?test|yarn test|pnpm test|cargo (test|check)|go (test|vet)|"
    r"tsc|mypy|ruff|eslint|flake8|make test|jest|vitest|python -m py_compile)\b"
)
ERROR_RE = re.compile(r"^(Error|Traceback)|\bTraceback \(most recent call last\)|No such file", re.M)


def load_config(name: str) -> dict:
    for p in sorted((HERE / "configs").glob("*.json")):
        cfg = json.loads(p.read_text())
        if name in (cfg["name"], cfg["tier"]):
            return cfg
    raise SystemExit(f"No config for '{name}' in {HERE / 'configs'}")


def iter_records(inputs: list[Path]):
    for inp in inputs:
        files = sorted(inp.glob("*.jsonl")) if inp.is_dir() else [inp]
        for f in files:
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def clean(record: dict) -> dict | None:
    """Validate and normalize one trajectory. None = drop it."""
    msgs = record.get("messages") or []
    tools = record.get("tools") or []
    tool_names = {t.get("function", {}).get("name") for t in tools}
    if len(msgs) < 3 or msgs[0].get("role") != "system":
        return None
    if msgs[-1].get("role") != "assistant" or not (msgs[-1].get("content") or "").strip():
        return None  # must end in a final answer

    pending: set[str] = set()
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            if pending:
                return None  # previous tool calls never got results
            calls = m.get("tool_calls") or []
            for tc in calls:
                fn = tc.get("function", {})
                if fn.get("name") not in tool_names:
                    return None
                try:
                    json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    return None
                if not tc.get("id"):
                    return None
                pending.add(tc["id"])
            m = {"role": "assistant", "content": m.get("content") or "",
                 **({"tool_calls": calls} if calls else {})}
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending:
                return None
            pending.discard(tid)
            content = str(m.get("content") or "")
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + "\n... [truncated]"
            m = {"role": "tool", "tool_call_id": tid, "content": content}
        elif role in ("system", "user"):
            m = {"role": role, "content": m.get("content") or ""}
        else:
            return None
        out.append(m)
    if pending:
        return None

    # Only ship the tools the trajectory could have used; MCP tools from one
    # user's setup shouldn't leak into everyone's training set.
    tools = [t for t in tools if not t.get("function", {}).get("name", "").startswith("mcp__")]
    return {"messages": out, "tools": tools, "meta": dict(record.get("meta") or {})}


# ── Features ──────────────────────────────────────────────────────────────────

def tool_calls(rec: dict) -> list[dict]:
    return [tc["function"] for m in rec["messages"] for tc in (m.get("tool_calls") or [])]


def n_tool_calls(rec: dict) -> int:
    return len(tool_calls(rec))


def n_chars(rec: dict) -> int:
    return sum(len(str(m.get("content") or "")) for m in rec["messages"])


def effort_for(rec: dict) -> str:
    n = n_tool_calls(rec)
    for limit, level in EFFORT_BY_TOOL_CALLS:
        if n <= limit:
            return level
    return "high"


def is_research(rec: dict) -> bool:
    """Read/search-only tool use that ends in a substantial written answer."""
    calls = tool_calls(rec)
    if not calls or any(c["name"] not in READ_ONLY_TOOLS for c in calls):
        return False
    return len(rec["messages"][-1].get("content") or "") >= 800


def reward(rec: dict) -> float:
    """Heuristic quality score in [0, 1]. Transparent on purpose — tune it.

      0.5  base: trajectory is valid and ends in an answer (clean() checked)
     +0.2  verified: ran tests/linters/type-checks after its last edit
     +0.1  substantive final answer (> 80 chars)
     -0.1  per tool error it hit (max -0.3)
     -0.2  looping: the same tool call issued 3+ times
    """
    score = 0.5
    msgs = rec["messages"]
    last_edit = max((i for i, m in enumerate(msgs) for tc in (m.get("tool_calls") or [])
                     if tc["function"]["name"] in ("edit_file", "write_file")), default=-1)
    if last_edit >= 0 and any(
        tc["function"]["name"] == "run_command" and VERIFY_RE.search(tc["function"]["arguments"])
        for m in msgs[last_edit + 1:] for tc in (m.get("tool_calls") or [])
    ):
        score += 0.2
    if len(msgs[-1].get("content") or "") > 80:
        score += 0.1
    errors = sum(1 for m in msgs if m["role"] == "tool" and ERROR_RE.search(m["content"] or ""))
    score -= min(0.3, 0.1 * errors)
    seen: dict[str, int] = {}
    for c in tool_calls(rec):
        key = c["name"] + c.get("arguments", "")
        seen[key] = seen.get(key, 0) + 1
    if any(v >= 3 for v in seen.values()):
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 3)


# ── Per-model slices ──────────────────────────────────────────────────────────

def fits(rec: dict, data_cfg: dict) -> bool:
    mt, mc = data_cfg.get("max_tool_calls"), data_cfg.get("max_chars")
    if mt is not None and n_tool_calls(rec) > mt:
        return False
    if mc is not None and n_chars(rec) > mc:
        return False
    if n_tool_calls(rec) < data_cfg.get("min_tool_calls", 0):
        return False
    if data_cfg.get("research_only") and not is_research(rec):
        return False
    return True


def with_effort_tag(rec: dict) -> dict:
    """Copy of rec with the effort tag appended to the system prompt."""
    msgs = [dict(m) for m in rec["messages"]]
    level = effort_for(rec)
    msgs[0]["content"] = (msgs[0]["content"] or "") + EFFORT_SUFFIX.format(effort=level)
    return {**rec, "messages": msgs, "meta": {**rec["meta"], "effort": level}}


def build_slice(recs: list[dict], cfg: dict) -> list[dict]:
    data_cfg = cfg.get("data", {})
    subset = [r for r in recs if fits(r, data_cfg)]
    if data_cfg.get("effort_tags"):
        subset = [with_effort_tag(r) for r in subset]
    return subset


def split_by_project(recs: list[dict], eval_frac: float, seed: int):
    projects = sorted({r["meta"].get("project", "unknown") for r in recs})
    rng = random.Random(seed)
    rng.shuffle(projects)
    n_eval = max(1, round(len(projects) * eval_frac)) if len(projects) > 1 else 0
    eval_projects = set(projects[:n_eval])
    train = [r for r in recs if r["meta"].get("project", "unknown") not in eval_projects]
    ev = [r for r in recs if r["meta"].get("project", "unknown") in eval_projects]
    if not ev and len(train) > 10:  # single project: fall back to a random split
        rng.shuffle(train)
        k = max(1, int(len(train) * eval_frac))
        ev, train = train[:k], train[k:]
    return train, ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", type=Path, help="dir or .jsonl (repeatable)")
    ap.add_argument("--model", action="append", choices=MODELS, help="default: all three")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--min-reward", type=float, default=0.5)
    ap.add_argument("--top", type=int, help="keep only the N highest-reward trajectories per model")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    inputs = [p.expanduser() for p in (args.input or [DEFAULT_INPUT])]
    seen: set[str] = set()
    recs: list[dict] = []
    dropped = low_reward = 0
    for raw in iter_records(inputs):
        rec = clean(raw)
        if rec is None:
            dropped += 1
            continue
        h = hashlib.sha256(json.dumps(rec["messages"][1:], sort_keys=True).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rec["meta"]["reward"] = reward(rec)
        if rec["meta"]["reward"] < args.min_reward:
            low_reward += 1
            continue
        recs.append(rec)
    recs.sort(key=lambda r: r["meta"]["reward"], reverse=True)
    print(f"{len(recs)} usable trajectories ({dropped} malformed, {low_reward} below reward "
          f"{args.min_reward}) from {', '.join(map(str, inputs))}")
    if not recs:
        raise SystemExit("Nothing to train on. Turn on training.log_trajectories or run generate_data.py.")

    for name in args.model or MODELS:
        cfg = load_config(name)
        subset = build_slice(recs, cfg)[: args.top]
        train, ev = split_by_project(subset, args.eval_frac, args.seed)
        out_dir = args.out / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in (("train", train), ("eval", ev)):
            with (out_dir / f"{split}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps({"messages": r["messages"], "tools": r["tools"]}) + "\n")
        extra = ""
        if cfg.get("data", {}).get("effort_tags"):
            counts = {lvl: sum(r["meta"]["effort"] == lvl for r in subset) for lvl in ("low", "medium", "high")}
            extra = "  effort " + " ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {cfg['name']:<10} train={len(train):>6}  eval={len(ev):>5}{extra}  → {out_dir}")
        if not subset and cfg.get("data", {}).get("research_only"):
            print("             (no research trajectories yet: add research tasks to generate_data.py)")


if __name__ == "__main__":
    main()
