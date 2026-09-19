from pathlib import Path


def load_skill(path: str) -> str:
    skill_path = Path(path)

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill not found: {skill_path}")

    content = skill_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Skill is empty: {skill_path}")

    return content