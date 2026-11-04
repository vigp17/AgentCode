#!/usr/bin/env python3
"""
Bootstrap training data by running AgentCode itself on a list of tasks.

    python finetune/generate_data.py --tasks finetune/tasks.jsonl \\
        --teacher <model> --out finetune/raw --workers 4

finetune/tasks.jsonl (~400 tasks, from make_tasks.py) is the full set;
tasks.example.jsonl is a 13-task sample for a quick trial. A task counts as
done only when it produced a training trajectory (saved to
<out>/trajectories.jsonl, tagged with the task's id), so re-running the same
command resumes where it stopped (Ctrl-C, crash, laptop sleep) and retries
everything else. Each failed task's output is saved to <out>/logs/<id>.log.

Each line of the tasks file is {"repo": "<path or git URL>", "task": "<prompt>"}.
For every task the repo is copied into a throwaway directory, AgentCode runs
the task non-interactively with --auto-approve, and the finished trajectory is
logged (same format as settings.training.log_trajectories). Feed the output
dir to prepare_data.py --input.

⚠  Runs agent-written shell commands with no prompts. Use a container or VM.

⚠  Teacher choice: most closed-model terms of service (Anthropic, OpenAI,
   Google) restrict using their outputs to train other models. Use an
   open-weight teacher whose license allows it (e.g. a large Qwen3-Coder or
   DeepSeek model via Together/Fireworks/your own vLLM — any litellm model
   string works), or check your agreement with the provider first.
"""

from __future__ import annotations

import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def task_id(task: dict) -> str:
    return hashlib.sha256(f"{task['repo']}\n{task['task']}".encode()).hexdigest()[:16]


def checkout(repo: str, dest: Path) -> None:
    if repo.startswith(("http://", "https://", "git@")):
        subprocess.run(["git", "clone", "--depth", "1", "-q", repo, str(dest)], check=True)
    else:
        shutil.copytree(Path(repo).expanduser(), dest, symlinks=True,
                        ignore=shutil.ignore_patterns(".agentcode*", "node_modules", ".venv"))


def last_error(text: str) -> str:
    """The most informative line of a failed run's output."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if any(k in ln for k in ("Error", "Exception", "error:", "Forbidden", "Unauthorized",
                                 "credits", "not found", "Timeout")):
            return ln[:300]
    return lines[-1][:300] if lines else "no output"


def check_teacher(teacher: str) -> None:
    """One tiny request before running hundreds of tasks: fail fast on a bad
    key, wrong model name or empty balance, with the provider's own message."""
    try:
        import litellm
        litellm.completion(model=teacher, messages=[{"role": "user", "content": "Say OK."}],
                           max_tokens=5)
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:400] if str(e).strip() else type(e).__name__
        raise SystemExit(
            f"Teacher check failed for '{teacher}':\n  {msg}\n\n"
            "Common causes: API key missing or wrong (is OPENROUTER_API_KEY passed into the\n"
            "container?), no credits on the account, or a typo in the model name."
        )
    print(f"Teacher OK: {teacher}")


def task_env(tmp: Path, repo: str, use_venv: bool = True, with_pip: bool = True) -> dict:
    """Environment for one task's agent.

    Each task gets its own Python venv first on PATH, so `pip install ...`
    run by the agent lands there and is thrown away with the task. Without
    this, the agent can overwrite or delete packages AgentCode itself needs:
    `pip install -e .` in a jinja checkout replaced jinja2 with a link into a
    temp folder, and every later task crashed once that folder was deleted.
    """
    env = {
        **os.environ,
        "AGENTCODE_TRAJECTORY_PROJECT": hashlib.sha256(repo.encode()).hexdigest()[:12],
        "AGENTCODE_DISABLE_DISCOVERY": "1",
    }
    env.pop("PYTHONPATH", None)
    if use_venv:
        venv = tmp / "venv"
        cmd = [sys.executable, "-m", "venv", str(venv)]
        if not with_pip:
            cmd.insert(3, "--without-pip")
        subprocess.run(cmd, check=True, capture_output=True)
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


STEP_LIMIT_EXIT = 3  # cli.py exits with this when it hits max_iterations


def done_tasks(out_dir: Path) -> set[str]:
    """Task ids that already have a trajectory in out_dir."""
    done: set[str] = set()
    for f in out_dir.glob("*.jsonl"):
        for line in f.open():
            try:
                tid = json.loads(line).get("meta", {}).get("task")
            except json.JSONDecodeError:
                continue
            if tid:
                done.add(tid)
    return done


def run_task(task: dict, teacher: str, timeout: int, max_iter: int,
             use_venv: bool = True) -> tuple[list[str], str, str]:
    """Run one task. Returns (trajectory lines, full output, failure reason)."""
    repo, prompt = task["repo"], task["task"]
    with tempfile.TemporaryDirectory(prefix="agentcode-gen-") as tmp:
        work = Path(tmp) / "repo"
        traj_dir = Path(tmp) / "traj"   # this task's own trajectories
        checkout(repo, work)
        (work / ".agentcode").mkdir(exist_ok=True)
        (work / ".agentcode" / "settings.json").write_text(json.dumps({
            "model": {"routing": False, "auto_discover": False},
            "training": {"log_trajectories": True, "trajectories_dir": str(traj_dir)},
            "limits": {"max_iterations": max_iter},
        }))
        env = task_env(Path(tmp), repo, use_venv)
        env["AGENTCODE_TRAJECTORY_TASK"] = task_id(task)
        # -I: AgentCode's own interpreter ignores PYTHONPATH / user site, so
        # nothing the agent sets up can shadow AgentCode's dependencies.
        cmd = [sys.executable, "-I", str(REPO_ROOT / "cli.py"), "-y", "--no-route",
               "--model", teacher, "--dir", str(work), prompt]
        try:
            r = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return [], "", f"timed out after {timeout}s (raise --timeout)"
        output = f"$ {' '.join(cmd[:6])} ...\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        lines = [ln for f in sorted(traj_dir.glob("*.jsonl")) for ln in f.read_text().splitlines()
                 if ln.strip()]
        if lines:
            return lines, output, ""
        if r.returncode == STEP_LIMIT_EXIT:
            return [], output, f"hit the {max_iter}-step limit before finishing (raise --max-iterations)"
        if r.returncode == 0:
            return [], output, "finished without a final answer, so no trajectory was saved"
        return [], output, last_error(output)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--teacher", required=True, help="litellm model string for the teacher")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "raw")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per task")
    ap.add_argument("--max-iterations", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1, help="tasks run in parallel")
    ap.add_argument("--limit", type=int, help="only run the first N pending tasks (trial run)")
    ap.add_argument("--skip-check", action="store_true", help="don't test the teacher first")
    ap.add_argument("--no-task-venv", action="store_true",
                    help="don't give each task its own Python venv (not recommended)")
    args = ap.parse_args()
    if not args.skip_check:
        check_teacher(args.teacher)

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.open() if line.strip()]
    done = done_tasks(args.out)
    pending = [t for t in tasks if task_id(t) not in done][: args.limit]
    n_done = sum(task_id(t) in done for t in tasks)
    print(f"{len(tasks)} tasks, {n_done} already done, running {len(pending)} "
          f"with {args.workers} worker(s)")

    lock = threading.Lock()
    ok = 0

    log_dir = args.out / "logs"

    out_file = args.out / "trajectories.jsonl"

    def one(task: dict) -> tuple[bool, str]:
        lines, output, why = run_task(task, args.teacher, args.timeout, args.max_iterations,
                                      use_venv=not args.no_task_venv)
        if lines:
            with lock, out_file.open("a") as f:
                f.write("\n".join(lines) + "\n")
            return True, ""
        # No trajectory → not done, so the next run retries it.
        log_dir.mkdir(exist_ok=True)
        (log_dir / f"{task_id(task)}.log").write_text(f"{task['repo']}\n{task['task']}\n\n{output}")
        return False, why

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one, t): t for t in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                good, why = fut.result()
            except Exception as e:  # clone failure etc. — keep going
                good, why = False, f"{type(e).__name__}: {str(e)[:200]}"
            ok += good
            print(f"[{i}/{len(pending)}] {'✓' if good else '✗'} "
                  f"[{task.get('category', '?')}] {task['task'][:64]}")
            if not good:
                print(f"      ↳ {why}   (log: {log_dir / (task_id(task) + '.log')})")
    print(f"\n{ok}/{len(pending)} tasks finished. Trajectories in {args.out}")
    print(f"Next: python finetune/prepare_data.py --input {args.out}")


if __name__ == "__main__":
    main()
