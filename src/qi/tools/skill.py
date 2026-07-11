from typing import Any

from pydantic import BaseModel

from qi.lib.skills import discover_skills, load_skill_body


class SkillParams(BaseModel):
    name: str


class SkillTool:
    name = "Skill"
    description = (
        "Load the full instructions for a named skill listed in AVAILABLE_SKILLS. "
        "Returns the skill's markdown body and its directory path; read supporting "
        "files there with ReadFile or Bash."
    )
    params = SkillParams

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

    def __call__(self, name: str) -> str:
        skills = discover_skills()
        skill = skills.get(name)
        if skill is None:
            available = ", ".join(sorted(skills)) or "(none)"
            return f"ERROR: Unknown skill '{name}'. Available skills: {available}"
        return f"Skill: {skill.name}\nDirectory: {skill.path}\n\n{load_skill_body(skill)}"
