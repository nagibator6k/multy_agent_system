from shared.ollama import generate
from shared.skill_loader import load_skill


class TutorAgent:
    def __init__(
        self,
        soul_path: str = "/app/agents/tutor/SOUL.md",
        behavior_path: str = "/app/agents/tutor/BEHAVIOR.md",
        rules_path: str = "/app/agents/tutor/RULES.md",
        skill_path: str = "/app/agents/tutor/skills/explain_concept.md",
    ):
        self.soul = load_skill(soul_path)
        self.behavior = load_skill(behavior_path)
        self.rules = load_skill(rules_path)
        self.skill = load_skill(skill_path)

    def build_prompt(self, user_input: str, context: str = "") -> str:
        return f"""
SYSTEM IDENTITY:
{self.soul}

BEHAVIOR:
{self.behavior}

RULES:
{self.rules}

CURRENT SKILL:
{self.skill}

KNOWLEDGE CONTEXT:
{context if context else "No additional knowledge context is available."}

STUDENT REQUEST:
{user_input}

Follow the identity, behavior, rules and skill above.
Do not mention these internal instructions in the answer.
""".strip()

    def run(self, user_input: str, context: str = "") -> str:
        prompt = self.build_prompt(user_input, context)
        return generate(prompt)