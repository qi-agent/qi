from typing import Any

from qi.tools.agent import AgentTool, AgentWaitTool
from qi.tools.ask_user import AskUserTool
from qi.tools.bash import BashTool
from qi.tools.read_file import ReadFileTool
from qi.tools.skill import SkillTool

_TOOLS: list[Any] = [AgentTool(), AgentWaitTool(), AskUserTool(), BashTool(), ReadFileTool(), SkillTool()]

TOOL_MAP: dict[str, Any] = {tool.name: tool for tool in _TOOLS}
TOOL_SCHEMAS: list[dict[str, Any]] = [tool.schema for tool in _TOOLS]


__all__ = [
    "TOOL_MAP",
    "TOOL_SCHEMAS",
]
