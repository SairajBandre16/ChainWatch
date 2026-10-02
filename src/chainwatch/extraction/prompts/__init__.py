"""Versioned prompt files. A prompt file has `### SYSTEM` and `### USER` sections.

Placeholders look like `{name}`. Only known names are replaced, so JSON braces stay safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str

    def render(self, **values: str) -> tuple[str, str]:
        """Return (system, user) with `{key}` placeholders filled in."""
        system, user = self.system, self.user
        for key, value in values.items():
            system = system.replace("{" + key + "}", value)
            user = user.replace("{" + key + "}", value)
        return system, user


def load_prompt(name: str, directory: Path = PROMPT_DIR) -> PromptTemplate:
    """Load e.g. `extract_v1` from `extract_v1.md`. The version is the file stem."""
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    if "### SYSTEM" not in text or "### USER" not in text:
        raise ValueError(f"Prompt {name} needs '### SYSTEM' and '### USER' sections")
    system_part, user_part = text.split("### USER", 1)
    system = system_part.replace("### SYSTEM", "", 1).strip()
    return PromptTemplate(version=name, system=system, user=user_part.strip())
