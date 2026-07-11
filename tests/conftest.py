"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the user config dir at a tmp dir so a developer's real
    ~/.config/qi (e.g. skills) can't leak into tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
