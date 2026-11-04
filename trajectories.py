"""
AgentCode — trajectory logging for fine-tuning.

When settings.training.log_trajectories is true, every agent turn that ends in
a final answer is appended to ~/.agentcode/trajectories/YYYY-MM-DD.jsonl in
the OpenAI chat fine-tuning format (messages + tools). finetune/prepare_data.py
turns these into train/eval splits for the agentcode-light/medium/heavy models.

Turns that hit max_iterations or raised errors are not logged — we only want
to teach the models trajectories that actually finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _dir(settings) -> Path:
    custom = getattr(getattr(settings, "training", None), "trajectories_dir", None)
    return Path(custom).expanduser() if custom else Path.home() / ".agentcode" / "trajectories"


def enabled(settings) -> bool:
    return bool(settings and getattr(settings.training, "log_trajectories", False))


def turn_slice(messages: list[dict], user_msg: dict) -> list[dict]:
    """Messages from this turn's user message to the end (identity match, so
    compaction rewriting older history doesn't break it)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i] is user_msg:
            return messages[i:]
    return []


def log_turn(
    settings,
    system: str,
    turn_messages: list[dict],
    tools: list[dict],
    model: str,
    project_dir: str,
) -> Path | None:
    """Append one finished turn. Never raises — logging must not break a turn."""
    if not enabled(settings) or not turn_messages:
        return None
    try:
        out_dir = _dir(settings)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        record = {
            "messages": [{"role": "system", "content": system}, *turn_messages],
            "tools": tools,
            "meta": {
                "model": model,
                "ts": time.time(),
                # Hash, not path: lets prepare_data split by project without
                # leaking directory names into a dataset you might share.
                # generate_data.py runs each task in a temp copy of a repo and
                # sets AGENTCODE_TRAJECTORY_PROJECT so copies share one id.
                "project": os.environ.get("AGENTCODE_TRAJECTORY_PROJECT")
                or hashlib.sha256(project_dir.encode()).hexdigest()[:12],
                "tool_calls": sum(len(m.get("tool_calls") or []) for m in turn_messages),
            },
        }
        with _lock, path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return path
    except Exception:
        return None
