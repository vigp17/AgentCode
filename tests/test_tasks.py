"""Training task list + generate_data.py resume logic (no agent runs)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
import generate_data  # noqa: E402
import make_tasks  # noqa: E402


def test_task_list_size_and_mix():
    tasks = make_tasks.build(seed=17)
    assert 300 <= len(tasks) <= 600
    cats = {t["category"] for t in tasks}
    assert cats == {"coding", "research", "long"}
    research = [t for t in tasks if t["category"] == "research"]
    assert len(research) >= 100                       # enough for veda-1
    assert not any(p in t["task"] for t in tasks for p in ("{test}", "{types}"))  # all filled in
    assert len({(t["repo"], t["task"]) for t in tasks}) == len(tasks)  # no duplicates


def test_research_tasks_are_read_only():
    for t in make_tasks.build(seed=17):
        if t["category"] == "research" and t["effort"] != "low":
            assert "read-only" in t["task"].lower() or "don't implement" in t["task"].lower(), t


def test_committed_tasks_file_is_current():
    committed = [json.loads(l) for l in (make_tasks.HERE / "tasks.jsonl").open()]
    assert committed == make_tasks.build(seed=17), "re-run: python finetune/make_tasks.py"


def _traj(task):
    tid = generate_data.task_id(task)
    return json.dumps({"messages": [], "tools": [], "meta": {"task": tid}})


def test_generate_data_resumes(tmp_path, monkeypatch):
    tasks = tmp_path / "t.jsonl"
    task_list = [{"repo": "r", "task": f"task {i}"} for i in range(5)]
    tasks.write_text("\n".join(json.dumps(t) for t in task_list))
    ran = []

    def fake_run(t, *a, **k):
        ran.append(t["task"])
        return [_traj(t)], "output", ""

    monkeypatch.setattr(generate_data, "run_task", fake_run)
    out = tmp_path / "o"
    argv = ["generate_data.py", "--tasks", str(tasks), "--teacher", "x", "--out", str(out),
            "--skip-check"]

    monkeypatch.setattr(sys, "argv", argv + ["--limit", "2"])
    generate_data.main()
    assert len(ran) == 2
    assert len(generate_data.done_tasks(out)) == 2   # done = has a trajectory

    monkeypatch.setattr(sys, "argv", argv + ["--workers", "3"])
    generate_data.main()  # resumes: only the 3 not yet done
    assert len(ran) == 5 and len(set(ran)) == 5
    assert len((out / "trajectories.jsonl").read_text().splitlines()) == 5


def test_clean_exit_without_trajectory_is_a_failure(tmp_path, monkeypatch):
    """The trial's bug: exit code 0 but no trajectory was counted as success."""
    tasks = tmp_path / "t.jsonl"
    tasks.write_text(json.dumps({"repo": "r", "task": "long one"}))
    out = tmp_path / "o"
    monkeypatch.setattr(sys, "argv", ["generate_data.py", "--tasks", str(tasks), "--teacher", "x",
                                      "--out", str(out), "--skip-check"])
    monkeypatch.setattr(generate_data, "run_task",
                        lambda *a, **k: ([], "out", "hit the 50-step limit before finishing"))
    generate_data.main()
    assert generate_data.done_tasks(out) == set()            # retried next run
    assert len(list((out / "logs").glob("*.log"))) == 1


def test_failed_tasks_are_logged_and_retried(tmp_path, monkeypatch):
    tasks = tmp_path / "t.jsonl"
    tasks.write_text(json.dumps({"repo": "r", "task": "flaky"}))
    out = tmp_path / "o"
    argv = ["generate_data.py", "--tasks", str(tasks), "--teacher", "x", "--out", str(out), "--skip-check"]
    monkeypatch.setattr(sys, "argv", argv)
    output = "--- stderr ---\nAuthenticationError: invalid key"
    monkeypatch.setattr(generate_data, "run_task",
                        lambda *a, **k: ([], output, generate_data.last_error(output)))
    generate_data.main()
    logs = list((out / "logs").glob("*.log"))
    assert len(logs) == 1 and "invalid key" in logs[0].read_text()
    assert generate_data.done_tasks(out) == set()
    assert generate_data.last_error("x\nAuthenticationError: invalid key\n") == \
        "AuthenticationError: invalid key"


def test_trajectory_records_task_id(tmp_path, monkeypatch):
    import trajectories
    from settings import Settings, TrainingSettings
    monkeypatch.setenv("AGENTCODE_TRAJECTORY_TASK", "abc123")
    s = Settings(training=TrainingSettings(log_trajectories=True, trajectories_dir=str(tmp_path)))
    path = trajectories.log_turn(s, "sys", [{"role": "user", "content": "x"},
                                            {"role": "assistant", "content": "y"}], [], "m", "/p")
    assert json.loads(path.read_text())["meta"]["task"] == "abc123"
    assert generate_data.done_tasks(tmp_path) == {"abc123"}


def test_teacher_check_stops_early(monkeypatch):
    import litellm
    def boom(**kw):
        raise RuntimeError("OpenrouterException - User not found.")
    monkeypatch.setattr(litellm, "completion", boom)
    try:
        generate_data.check_teacher("openrouter/x")
    except SystemExit as e:
        assert "User not found" in str(e) and "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_each_task_gets_its_own_python(tmp_path, monkeypatch):
    import os
    import subprocess
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = generate_data.task_env(tmp_path, "https://github.com/pallets/jinja", with_pip=False)
    venv_bin = str(tmp_path / "venv" / "bin")
    assert env["PATH"].split(os.pathsep)[0] == venv_bin
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")
    assert "PYTHONPATH" not in env
    # `python` for the agent's commands resolves to the task venv, not ours.
    out = subprocess.run(["python", "-c", "import sys; print(sys.prefix)"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(tmp_path / "venv")


def test_agentcode_starts_with_the_flags_generate_data_uses(tmp_path):
    """Launch cli.py exactly as run_task does (from another folder) — the
    `python -I` regression made every task fail with No module named 'agent'."""
    import subprocess
    r = subprocess.run([sys.executable, *generate_data.PY_FLAGS,
                        str(generate_data.REPO_ROOT / "cli.py"), "--help"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
