from typing import Any

from qi.tools.bash import BashTool
from qi.tools.read_file import ReadFileTool

_TOOLS: list[Any] = [BashTool(), ReadFileTool()]

TOOL_MAP: dict[str, Any] = {tool.name: tool for tool in _TOOLS}
TOOL_SCHEMAS: list[dict[str, Any]] = [tool.schema for tool in _TOOLS]


__all__ = [
    "TOOL_MAP",
    "TOOL_SCHEMAS",
]
