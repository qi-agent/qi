"""Shared test fixtures."""

from collections.abc import Iterator

import pytest

from qi.lib import agents


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the user config dir at a tmp dir so a developer's real
    ~/.config/qi (e.g. skills) can't leak into tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))


@pytest.fixture(autouse=True)
def reset_agent_runtime() -> Iterator[None]:
    """Tear down the module-level agent runner so no test leaks subagent
    processes or an event sink into the next one."""
    yield
    agents.reset_runner()
    agents.set_event_sink(None)
