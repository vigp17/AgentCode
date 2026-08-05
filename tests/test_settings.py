"""Tests for settings layering: defaults → global → project → CLI."""

import json

import settings as settings_mod
from settings import DEFAULT_AUTO_APPROVE, generate_starter_settings, load_settings


def _write_settings(directory, data):
    d = directory / ".agentcode"
    d.mkdir(exist_ok=True)
    (d / "settings.json").write_text(json.dumps(data))


def test_defaults_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: tmp_path / "home"))

    s = load_settings(str(tmp_path))

    assert s.model.default == "claude-sonnet-5"
    assert s.model.routing is True
    assert s.permissions.auto_approve == DEFAULT_AUTO_APPROVE
    assert s.limits.max_iterations == 25


def test_project_overrides_global(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".agentcode").mkdir(parents=True)
    (home / ".agentcode" / "settings.json").write_text(
        json.dumps({"model": {"default": "from-global", "routing": False}})
    )
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: home))

    project = tmp_path / "proj"
    project.mkdir()
    _write_settings(project, {"model": {"default": "from-project"}})

    s = load_settings(str(project))

    assert s.model.default == "from-project"
    # Untouched keys still come from the global layer.
    assert s.model.routing is False


def test_cli_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    _write_settings(tmp_path, {"permissions": {"auto_approve_all": False}})

    s = load_settings(str(tmp_path), {"permissions": {"auto_approve_all": True}})

    assert s.permissions.auto_approve_all is True


def test_deny_list_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    _write_settings(tmp_path, {"permissions": {"deny": ["run_command", "git_push"]}})

    s = load_settings(str(tmp_path))

    assert s.permissions.deny == ["run_command", "git_push"]


def test_malformed_json_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    d = tmp_path / ".agentcode"
    d.mkdir()
    (d / "settings.json").write_text("{ not valid json")

    s = load_settings(str(tmp_path))

    assert s.model.default == "claude-sonnet-5"


def test_generate_starter_settings_is_idempotent(tmp_path):
    path, created = generate_starter_settings(str(tmp_path))
    assert created and path.exists()

    path2, created2 = generate_starter_settings(str(tmp_path))
    assert not created2 and path2 == path


def test_starter_settings_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    generate_starter_settings(str(tmp_path))

    s = load_settings(str(tmp_path))

    assert s.model.default == "claude-sonnet-5"
    assert "read_file" in s.permissions.auto_approve
