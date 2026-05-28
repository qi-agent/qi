from qi.tools.read_file import ReadFileTool

_TOOLS = [ReadFileTool()]

TOOL_MAP = {tool.NAME: tool for tool in _TOOLS}
TOOL_SCHEMAS = [tool.schema for tool in _TOOLS]


__all__ = [
    "TOOL_MAP",
    "TOOL_SCHEMAS",
]
