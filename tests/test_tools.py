"""Tests for the built-in tool implementations."""

import tools
from tools import execute_tool


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_returns_numbered_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("one\ntwo\nthree\n")

    out = execute_tool("read_file", {"path": str(f)})

    assert "one" in out and "three" in out
    assert "3 lines total" in out


def test_read_file_line_range(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)))

    out = execute_tool("read_file", {"path": str(f), "start_line": 2, "end_line": 4})

    assert "line2" in out and "line4" in out
    assert "line5" not in out


def test_read_file_missing(tmp_path):
    out = execute_tool("read_file", {"path": str(tmp_path / "nope.py")})
    assert "not found" in out.lower()


def test_read_file_respects_size_limit(tmp_path, monkeypatch):
    monkeypatch.setitem(tools._LIMITS, "max_file_size", 10)
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)

    out = execute_tool("read_file", {"path": str(f)})

    assert "too large" in out.lower()


# ── write_file / edit_file ────────────────────────────────────────────────────

def test_write_file_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "a.txt"

    out = execute_tool("write_file", {"path": str(target), "content": "hi"})

    assert target.read_text() == "hi"
    assert "✓" in out


def test_edit_file_replaces_unique_string(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("alpha\nbeta\ngamma\n")

    out = execute_tool(
        "edit_file", {"path": str(f), "old_string": "beta", "new_string": "BETA"}
    )

    assert f.read_text() == "alpha\nBETA\ngamma\n"
    assert "✓" in out


def test_edit_file_refuses_ambiguous_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("dup\ndup\n")
    before = f.read_text()

    out = execute_tool(
        "edit_file", {"path": str(f), "old_string": "dup", "new_string": "x"}
    )

    assert "appears 2 times" in out
    assert f.read_text() == before


def test_edit_file_reports_missing_string(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("alpha\n")

    out = execute_tool(
        "edit_file", {"path": str(f), "old_string": "nope", "new_string": "x"}
    )

    assert "not found" in out


# ── search ────────────────────────────────────────────────────────────────────

def test_search_files_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")

    out = execute_tool("search_files", {"pattern": "*.py", "path": str(tmp_path)})

    assert "a.py" in out and "b.py" in out
    assert "c.txt" not in out


def test_search_files_no_match(tmp_path):
    out = execute_tool("search_files", {"pattern": "*.rs", "path": str(tmp_path)})
    assert "No files matching" in out


def test_search_text_finds_matches(tmp_path):
    (tmp_path / "a.py").write_text("import os\nNEEDLE here\n")

    out = execute_tool("search_text", {"pattern": "NEEDLE", "path": str(tmp_path)})

    assert "NEEDLE" in out
    assert "a.py" in out


# ── list_directory ────────────────────────────────────────────────────────────

def test_list_directory_shows_entries(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("x")

    out = execute_tool("list_directory", {"path": str(tmp_path)})

    assert "a.py" in out and "sub" in out


def test_list_directory_rejects_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x")
    assert "Not a directory" in execute_tool("list_directory", {"path": str(f)})


# ── dispatch ──────────────────────────────────────────────────────────────────

def test_unknown_tool():
    assert "Unknown tool" in execute_tool("no_such_tool", {})


def test_invalid_arguments_are_reported(tmp_path):
    out = execute_tool("read_file", {"wrong_kwarg": 1})
    assert "Invalid arguments" in out


def test_configure_limits_applies(monkeypatch):
    from settings import LimitsSettings

    original = dict(tools._LIMITS)
    try:
        tools.configure_limits(LimitsSettings(max_file_size=1, max_output=2, max_search_results=3))
        assert tools._LIMITS["max_file_size"] == 1
        assert tools._LIMITS["max_output"] == 2
        assert tools._LIMITS["max_search_results"] == 3
    finally:
        tools._LIMITS.update(original)
