from typing import Any

from pydantic import BaseModel


class ReadFileParams(BaseModel):
    path: str
    start: int = 0
    end: int | None = None


class ReadFileTool:
    name = "ReadFile"
    description = "Read contents of a file from the local filesystem by line range."
    params = ReadFileParams

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

    def __call__(self, path: str, start: int = 0, end: int | None = None) -> str:
        params = self.params(path=path, start=start, end=end)
        with open(params.path) as f:
            lines = f.readlines()
        if params.end is not None:
            lines = lines[params.start : params.end]
        elif params.start > 0:
            lines = lines[params.start :]
        return "".join(lines)
