from typing import Any

from pydantic import BaseModel

from qi.lib.agents import DEFAULT_TIMEOUT, get_runner


class AgentParams(BaseModel):
    prompt: str
    name: str | None = None
    files: list[str] = []
    reads_from: list[str] = []
    background: bool = False
    timeout: int = DEFAULT_TIMEOUT


class AgentTool:
    name = "Agent"
    description = (
        "Spawn a subagent: an independent qi process that completes a delegated task "
        "and returns its final reply. reads_from pipes the replies of previously "
        "spawned agents into this one, so fan-out and fan-in build a pipeline graph. "
        "background=true returns immediately so several agents run in parallel; "
        "collect their replies with AgentWait."
    )
    params = AgentParams

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params.model_json_schema(),
            },
        }

    def __call__(
        self,
        prompt: str,
        name: str | None = None,
        files: list[str] | None = None,
        reads_from: list[str] | None = None,
        background: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        params = self.params(
            prompt=prompt,
            name=name,
            files=files or [],
            reads_from=reads_from or [],
            background=background,
            timeout=timeout,
        )
        return get_runner().spawn(
            prompt=params.prompt,
            name=params.name,
            files=params.files,
            reads_from=params.reads_from,
            background=params.background,
            timeout=params.timeout,
        )


class AgentWaitParams(BaseModel):
    names: list[str] = []


class AgentWaitTool:
    name = "AgentWait"
    description = (
        "Wait for spawned subagents to finish and return their final replies. "
        "Omit names to wait for every agent spawned so far."
    )
    params = AgentWaitParams

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params.model_json_schema(),
            },
        }

    def __call__(self, names: list[str] | None = None) -> str:
        params = self.params(names=names or [])
        return get_runner().wait(params.names or None)
