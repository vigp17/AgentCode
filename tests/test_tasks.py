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


def test_generate_data_resumes(tmp_path, monkeypatch):
    tasks = tmp_path / "t.jsonl"
    tasks.write_text("\n".join(json.dumps({"repo": "r", "task": f"task {i}"}) for i in range(5)))
    ran = []
    monkeypatch.setattr(generate_data, "run_task", lambda t, *a: ran.append(t["task"]) or True)
    argv = ["generate_data.py", "--tasks", str(tasks), "--teacher", "x", "--out", str(tmp_path / "o")]

    monkeypatch.setattr(sys, "argv", argv + ["--limit", "2"])
    generate_data.main()
    assert len(ran) == 2

    monkeypatch.setattr(sys, "argv", argv + ["--workers", "3"])
    generate_data.main()  # resumes: only the 3 not yet done
    assert len(ran) == 5 and len(set(ran)) == 5
