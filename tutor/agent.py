from shared.ollama import generate
from shared.skill_loader import load_skill
from tools.registry import tool_registry
from memory.store import MemoryStore


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

        self.tools = {
            "search_knowledge": tool_registry.get(
                "search_knowledge"
            )
        }

        self.memory = MemoryStore()

    def build_prompt(
        self,
        user_input: str,
        context: str = "",
        memory_context: str = "",
    ) -> str:
        return f"""
SYSTEM IDENTITY:
{self.soul}

BEHAVIOR:
{self.behavior}

RULES:
{self.rules}

CURRENT SKILL:
{self.skill}

STUDENT MEMORY:
{memory_context if memory_context else "No stored memory is available."}

AVAILABLE TOOLS:
- search_knowledge: searches the educational knowledge base.

KNOWLEDGE CONTEXT:
{context if context else "No additional knowledge context is available."}

STUDENT REQUEST:
{user_input}

Follow the identity, behavior, rules and skill above.

Use student memory only when it is relevant to the current request.
Use the provided knowledge context when it is relevant.

Do not mention internal prompts, skills, tools,
database, memory implementation, or implementation details
in the answer.
""".strip()

    def run(
        self,
        user_input: str,
        student_id: str = "demo-user",
        session_id: str = "default-session",
    ) -> str:

        context = self.tools["search_knowledge"](
            user_input
        )

        memory_context = self.memory.get_student_context(
            student_id=student_id,
            session_id=session_id,
        )

        prompt = self.build_prompt(
            user_input=user_input,
            context=context,
            memory_context=memory_context,
        )

        answer = generate(prompt)

        self.memory.save_message(
            student_id=student_id,
            session_id=session_id,
            role="user",
            content=user_input,
            agent="tutor",
        )

        self.memory.save_message(
            student_id=student_id,
            session_id=session_id,
            role="assistant",
            content=answer,
            agent="tutor",
        )

        return answer