"""Tests for the init subcommand."""

from pathlib import Path

import pytest

from qi.commands.init import DEFAULT_CONFIG, run


def test_init_creates_dot_qi(tmp_path: Path) -> None:
    rc = run(["--dir", str(tmp_path)])
    assert rc == 0
    qi_dir = tmp_path / ".qi"
    assert qi_dir.is_dir()
    assert (qi_dir / "sessions").is_dir()
    config = qi_dir / "config.toml"
    assert config.read_text() == DEFAULT_CONFIG


def test_init_defaults_to_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rc = run([])
    assert rc == 0
    assert (tmp_path / ".qi" / "config.toml").exists()


def test_init_existing_config_fails(tmp_path: Path) -> None:
    qi_dir = tmp_path / ".qi"
    qi_dir.mkdir()
    (qi_dir / "config.toml").write_text("existing")
    rc = run(["--dir", str(tmp_path)])
    assert rc == 1
    assert (qi_dir / "config.toml").read_text() == "existing"


def test_init_existing_config_with_force(tmp_path: Path) -> None:
    qi_dir = tmp_path / ".qi"
    qi_dir.mkdir()
    (qi_dir / "config.toml").write_text("existing")
    rc = run(["--dir", str(tmp_path), "--force"])
    assert rc == 0
    assert (qi_dir / "config.toml").read_text() == DEFAULT_CONFIG


def test_init_preserves_existing_sessions(tmp_path: Path) -> None:
    sessions = tmp_path / ".qi" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "old.jsonl").write_text("data")
    rc = run(["--dir", str(tmp_path), "--force"])
    assert rc == 0
    assert (sessions / "old.jsonl").read_text() == "data"


def test_init_config_content(tmp_path: Path) -> None:
    rc = run(["--dir", str(tmp_path)])
    assert rc == 0
    content = (tmp_path / ".qi" / "config.toml").read_text()
    assert "gemma4:26b-mlx" in content
    assert "http://localhost:11434/v1" in content


def test_init_cli_routes_to_init(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from qi.cli import main
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.init.run", mock_run)
    rc = main(["init"])
    assert rc == 0
    mock_run.assert_called_once_with([])


def test_init_cli_passes_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from qi.cli import main
    mock_run = Mock(return_value=0)
    monkeypatch.setattr("qi.commands.init.run", mock_run)
    rc = main(["init", "--dir", "/some/path"])
    assert rc == 0
    mock_run.assert_called_once_with(["--dir", "/some/path"])
