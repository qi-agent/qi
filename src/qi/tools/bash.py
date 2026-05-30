import json
import logging
import subprocess
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class BashParams(BaseModel):
    command: str
    workdir: str | None = None
    timeout: int = 30


class BashTool:
    name = "Bash"
    description = "Execute a bash command or script and return its stdout, stderr, and exit code."
    params = BashParams

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
        self, command: str, workdir: str | None = None, timeout: int = 30
    ) -> str:
        params = self.params(command=command, workdir=workdir, timeout=timeout)
        logger.info(f"Running command: {params.command}")
        try:
            result = subprocess.run(
                params.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=params.workdir,
                timeout=params.timeout,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            timeout_msg = f"Command timed out after {params.timeout}s"
            return json.dumps({"exit_code": -1, "stdout": out, "stderr": f"{err}\n{timeout_msg}"})

        return json.dumps(
            {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
