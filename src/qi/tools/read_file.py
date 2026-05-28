from typing import Any

from pydantic import BaseModel


class ReadFileParams(BaseModel):
    path: str
    start: int = 0
    end: int | None = None


class ReadFileTool:
    NAME = "ReadFile"
    DESCRIPTION = "Read contents of a file from the local filesystem by line range."
    PARAMS = ReadFileParams

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": self.PARAMS.model_json_schema(),
            },
        }

    def __call__(self, path: str, start: int = 0, end: int | None = None) -> str:
        params = self.PARAMS(path=path, start=start, end=end)
        with open(params.path) as f:
            lines = f.readlines()
        if params.end is not None:
            lines = lines[params.start : params.end]
        elif params.start > 0:
            lines = lines[params.start :]
        return "".join(lines)
