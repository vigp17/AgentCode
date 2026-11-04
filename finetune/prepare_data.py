#!/usr/bin/env python3
"""
Turn AgentCode trajectories into train/eval sets for the three fine-tunes.

    python finetune/prepare_data.py                         # all three tiers
    python finetune/prepare_data.py --tier light
    python finetune/prepare_data.py --input ~/my/traj --input extra.jsonl

Input: JSONL files written by AgentCode when settings.training.log_trajectories
is on (default dir ~/.agentcode/trajectories), or by generate_data.py. Each line
is {"messages": [...], "tools": [...], "meta": {...}} in OpenAI chat format.

What it does:
  * drops turns whose tool calls are malformed (unknown tool, non-JSON args)
    or whose tool_call ids don't line up with tool results
  * truncates oversized tool results (huge file reads teach nothing)
  * de-duplicates identical trajectories
  * keeps per-tier slices: light learns short trajectories, heavy learns all
  * splits train/eval *by project* so eval measures generalization, not recall

Output: finetune/data/<model>/{train,eval}.jsonl  (stdlib only — runs anywhere)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = Path.home() / ".agentcode" / "trajectories"
MAX_TOOL_RESULT_CHARS = 6000


def load_config(name_or_tier: str) -> dict:
    for p in sorted((HERE / "configs").glob("*.json")):
        cfg = json.loads(p.read_text())
        if name_or_tier in (cfg["name"], cfg["tier"]):
            return cfg
    raise SystemExit(f"No config for '{name_or_tier}' in {HERE / 'configs'}")


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
    return {"messages": out, "tools": tools, "meta": record.get("meta", {})}


def n_tool_calls(rec: dict) -> int:
    return sum(len(m.get("tool_calls") or []) for m in rec["messages"])


def n_chars(rec: dict) -> int:
    return sum(len(str(m.get("content") or "")) for m in rec["messages"])


def fits_tier(rec: dict, data_cfg: dict) -> bool:
    mt = data_cfg.get("max_tool_calls")
    mc = data_cfg.get("max_chars")
    return (mt is None or n_tool_calls(rec) <= mt) and (mc is None or n_chars(rec) <= mc)


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
    ap.add_argument("--tier", action="append", help="light|medium|heavy (default: all)")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    inputs = [p.expanduser() for p in (args.input or [DEFAULT_INPUT])]
    seen: set[str] = set()
    recs: list[dict] = []
    dropped = 0
    for raw in iter_records(inputs):
        rec = clean(raw)
        if rec is None:
            dropped += 1
            continue
        h = hashlib.sha256(json.dumps(rec["messages"][1:], sort_keys=True).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        recs.append(rec)
    print(f"{len(recs)} usable trajectories ({dropped} dropped) from {', '.join(map(str, inputs))}")
    if not recs:
        raise SystemExit("Nothing to train on. Turn on training.log_trajectories or run generate_data.py.")

    for tier in args.tier or ["light", "medium", "heavy"]:
        cfg = load_config(tier)
        subset = [r for r in recs if fits_tier(r, cfg.get("data", {}))]
        train, ev = split_by_project(subset, args.eval_frac, args.seed)
        out_dir = args.out / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", train), ("eval", ev)):
            with (out_dir / f"{name}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps({"messages": r["messages"], "tools": r["tools"]}) + "\n")
        print(f"  {cfg['name']:<18} train={len(train):>6}  eval={len(ev):>5}  → {out_dir}")


if __name__ == "__main__":
    main()
