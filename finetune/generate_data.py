#!/usr/bin/env python3
"""
Bootstrap training data by running AgentCode itself on a list of tasks.

    python finetune/generate_data.py --tasks finetune/tasks.example.jsonl \\
        --teacher <model> --out finetune/raw

Each line of the tasks file is {"repo": "<path or git URL>", "task": "<prompt>"}.
For every task the repo is copied into a throwaway directory, AgentCode runs
the task non-interactively with --auto-approve, and the finished trajectory is
logged (same format as settings.training.log_trajectories). Feed the output
dir to prepare_data.py --input.

⚠  Runs agent-written shell commands with no prompts. Use a container or VM.

⚠  Teacher choice: most closed-model terms of service (Anthropic, OpenAI,
   Google) restrict using their outputs to train other models. Use an
   open-weight teacher whose license allows it (e.g. a large Qwen-Coder or
   DeepSeek model via Together/Fireworks/your own vLLM — any litellm model
   string works), or check your agreement with the provider first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def checkout(repo: str, dest: Path) -> None:
    if repo.startswith(("http://", "https://", "git@")):
        subprocess.run(["git", "clone", "--depth", "1", "-q", repo, str(dest)], check=True)
    else:
        shutil.copytree(Path(repo).expanduser(), dest, symlinks=True,
                        ignore=shutil.ignore_patterns(".agentcode*", "node_modules", ".venv"))


def run_task(task: dict, teacher: str, out_dir: Path, timeout: int, max_iter: int) -> bool:
    repo, prompt = task["repo"], task["task"]
    with tempfile.TemporaryDirectory(prefix="agentcode-gen-") as tmp:
        work = Path(tmp) / "repo"
        checkout(repo, work)
        (work / ".agentcode").mkdir(exist_ok=True)
        (work / ".agentcode" / "settings.json").write_text(json.dumps({
            "model": {"routing": False, "auto_discover": False},
            "training": {"log_trajectories": True, "trajectories_dir": str(out_dir)},
            "limits": {"max_iterations": max_iter},
        }))
        env = {
            **os.environ,
            "AGENTCODE_TRAJECTORY_PROJECT": hashlib.sha256(repo.encode()).hexdigest()[:12],
            "AGENTCODE_DISABLE_DISCOVERY": "1",
        }
        cmd = [sys.executable, str(REPO_ROOT / "cli.py"), "-y", "--no-route",
               "--model", teacher, "--dir", str(work), prompt]
        try:
            r = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--teacher", required=True, help="litellm model string for the teacher")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per task")
    ap.add_argument("--max-iterations", type=int, default=40)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.open() if line.strip()]
    ok = 0
    for i, task in enumerate(tasks, 1):
        good = run_task(task, args.teacher, args.out.resolve(), args.timeout, args.max_iterations)
        ok += good
        print(f"[{i}/{len(tasks)}] {'✓' if good else '✗'} {task['task'][:70]}")
    print(f"\n{ok}/{len(tasks)} tasks finished. Trajectories in {args.out}")
    print(f"Next: python finetune/prepare_data.py --input {args.out}")


if __name__ == "__main__":
    main()
