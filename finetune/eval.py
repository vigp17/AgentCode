#!/usr/bin/env python3
"""
Score a model on held-out AgentCode trajectories (next-action accuracy).

    python finetune/eval.py axon-1
    python finetune/eval.py axon-1 --baseline claude-haiku-4-5

For every assistant step in eval.jsonl the model sees the conversation up to
that point (teacher forcing) and must produce the same next action as the
reference: the same tool, or a final answer when the reference answered.

Reported:
  action_acc   predicted tool (or "answer") matches the reference
  valid_call   tool calls that name a real tool with JSON-parseable arguments
  args_match   for matching tools, the arguments' keys also match

Any model AgentCode can use works, including the fine-tunes (resolved through
settings / AGENTCODE_FT_*_API_BASE exactly like the CLI does).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import litellm  # noqa: E402

import finetuned  # noqa: E402
from settings import load_settings  # noqa: E402
from xml_tool_parser import (  # noqa: E402
    looks_like_json_tool_call, looks_like_xml_tool_call,
    parse_json_tool_calls, parse_xml_tool_calls,
)


def predict(model: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    resp = litellm.completion(
        **finetuned.completion_kwargs(model), messages=messages, tools=tools,
        tool_choice="auto", max_tokens=1024, temperature=0,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        return [{"name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                for tc in msg.tool_calls]
    text = msg.content or ""
    names = {t["function"]["name"] for t in tools}
    if looks_like_xml_tool_call(text):
        return parse_xml_tool_calls(text)[1]
    if looks_like_json_tool_call(text):
        return parse_json_tool_calls(text, names)[1]
    return []


def score(model: str, rows: list[dict], max_steps: int) -> dict:
    n = action_ok = calls = valid = arg_n = arg_ok = 0
    for r in rows:
        msgs, tools = r["messages"], r["tools"]
        names = {t["function"]["name"] for t in tools}
        steps = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][:max_steps]
        for i in steps:
            ref = msgs[i].get("tool_calls") or []
            try:
                pred = predict(model, msgs[:i], tools)
            except Exception as e:  # count as a miss, keep going
                print(f"  ! {type(e).__name__}: {str(e)[:120]}")
                pred = None
            n += 1
            if pred is None:
                continue
            ref_action = ref[0]["function"]["name"] if ref else "answer"
            pred_action = pred[0]["name"] if pred else "answer"
            action_ok += ref_action == pred_action
            for p in pred:
                calls += 1
                try:
                    json.loads(p["arguments"])
                    valid += p["name"] in names
                except (json.JSONDecodeError, TypeError):
                    pass
            if ref and pred and ref_action == pred_action:
                arg_n += 1
                try:
                    arg_ok += (set(json.loads(ref[0]["function"]["arguments"]))
                               == set(json.loads(pred[0]["arguments"])))
                except (json.JSONDecodeError, TypeError):
                    pass
    pct = lambda a, b: round(100 * a / b, 1) if b else None  # noqa: E731
    return {"model": model, "steps": n, "action_acc": pct(action_ok, n),
            "valid_call": pct(valid, calls), "args_match": pct(arg_ok, arg_n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--baseline", action="append", default=[], help="compare against (repeatable)")
    ap.add_argument("--data", type=Path, help="eval.jsonl (default: finetune/data/<model>/eval.jsonl)")
    ap.add_argument("--limit", type=int, default=100, help="max trajectories")
    ap.add_argument("--max-steps", type=int, default=6, help="max assistant steps per trajectory")
    args = ap.parse_args()

    finetuned.configure(load_settings(str(Path.cwd())).finetuned)
    data = args.data or HERE / "data" / args.model / "eval.jsonl"
    rows = [json.loads(line) for line in data.open() if line.strip()][: args.limit]
    print(f"{len(rows)} eval trajectories from {data}\n")
    print(f"{'model':<34}{'steps':>7}{'action_acc':>12}{'valid_call':>12}{'args_match':>12}")
    for model in [args.model, *args.baseline]:
        s = score(model, rows, args.max_steps)
        fmt = lambda v: "—" if v is None else f"{v}%"  # noqa: E731
        print(f"{model:<34}{s['steps']:>7}{fmt(s['action_acc']):>12}"
              f"{fmt(s['valid_call']):>12}{fmt(s['args_match']):>12}")


if __name__ == "__main__":
    main()
