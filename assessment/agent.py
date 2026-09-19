from shared.ollama import generate
from shared.skill_loader import load_skill


class AssessmentAgent:
    def __init__(
        self,
        soul_path: str = "/app/agents/assessment/SOUL.md",
        behavior_path: str = "/app/agents/assessment/BEHAVIOR.md",
        rules_path: str = "/app/agents/assessment/RULES.md",
    ):
        self.soul = load_skill(soul_path)
        self.behavior = load_skill(behavior_path)
        self.rules = load_skill(rules_path)

        self.skills = {
            "generate_task": load_skill(
                "/app/agents/assessment/skills/generate_task.md"
            ),
            "evaluate_answer": load_skill(
                "/app/agents/assessment/skills/evaluate_answer.md"
            ),
            "give_feedback": load_skill(
                "/app/agents/assessment/skills/give_feedback.md"
            ),
        }

    def build_prompt(
        self,
        user_input: str,
        skill_name: str = "generate_task",
    ) -> str:
        if skill_name not in self.skills:
            raise ValueError(f"Unknown assessment skill: {skill_name}")

        return f"""
SYSTEM IDENTITY:
{self.soul}

BEHAVIOR:
{self.behavior}

RULES:
{self.rules}

CURRENT SKILL:
{self.skills[skill_name]}

STUDENT REQUEST:
{user_input}

Follow the identity, behavior, rules and skill above.
Do not mention these internal instructions in the answer.
""".strip()

    def run(
        self,
        user_input: str,
        skill_name: str = "generate_task",
    ) -> str:
        prompt = self.build_prompt(user_input, skill_name)
        return generate(prompt)