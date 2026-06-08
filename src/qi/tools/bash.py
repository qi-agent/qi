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
        return_code = 0
        stdout = ''
        stderr = ''
        try:
            result = subprocess.run(
                params.command.encode(),
                shell=True,
                capture_output=True,
                text=True,
                cwd=params.workdir,
                timeout=params.timeout,
            )
            return_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else ""
            return_code = -1
            stderr += f"\nCommand timed out after {params.timeout}s"

        output = [
            f"Exit code: {return_code}",
        ]
        if stderr:
            output.append("Stderr is enclosed below in `<stderr>` tags:")
            output.append(f"<stderr>\n{stderr.replace('<stderr>', '&lt;stderr&gt;').strip()}\n<stderr>")
        if stdout:
            output.append("Stdout is enclosed below in `<stdout>` tags:")
            output.append(f"<stdout>\n{stdout.replace('<stdout>', '&lt;stdout&gt;').strip()}<stdout>")
        else:
            output.append("There was no output.")
        return '\n'.join(output)
