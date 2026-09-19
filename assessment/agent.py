import json
import re

from shared.ollama import generate
from shared.skill_loader import load_skill
from tools.registry import tool_registry
from tools.selector import ToolSelector


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

        self.tool_selector = ToolSelector()

    @staticmethod
    def _parse_number(
        value: str,
        default: float = 1.0,
    ) -> float:
        value = value.strip()

        if value in ("", "+"):
            return default

        if value == "-":
            return -default

        return float(value)

    @staticmethod
    def _extract_quadratic_coefficients(
        text: str,
    ) -> tuple[float, float, float] | None:

        normalized = text.lower()

        normalized = normalized.replace(
            "²",
            "^2",
        )

        normalized = normalized.replace(
            "−",
            "-",
        )

        normalized = normalized.replace(
            "–",
            "-",
        )

        normalized = normalized.replace(
            "—",
            "-",
        )

        normalized = re.sub(
            r"\s+",
            "",
            normalized,
        )

        pattern = re.compile(
            r"(?P<a>[+-]?\d*\.?\d*)x\^2"
            r"(?P<b>[+-]\d*\.?\d*)x"
            r"(?P<c>[+-]\d*\.?\d*)=0"
        )

        match = pattern.search(normalized)

        if not match:
            return None

        try:
            a = AssessmentAgent._parse_number(
                match.group("a"),
                default=1.0,
            )

            b = AssessmentAgent._parse_number(
                match.group("b"),
                default=1.0,
            )

            c = AssessmentAgent._parse_number(
                match.group("c"),
                default=0.0,
            )

            return a, b, c

        except ValueError:
            return None

    def _run_selected_tool(
        self,
        tool_name: str | None,
        user_input: str,
        skill_name: str,
    ) -> dict | None:

        if tool_name is None:
            return None

        if tool_name == "solve_quadratic":
            coefficients = (
                self._extract_quadratic_coefficients(
                    user_input
                )
            )

            if coefficients is None:
                return None

            a, b, c = coefficients

            result = tool_registry.call(
                "solve_quadratic",
                a=a,
                b=b,
                c=c,
            )

            return {
                "tool": "solve_quadratic",
                "input": {
                    "a": a,
                    "b": b,
                    "c": c,
                },
                "result": result,
            }

        if tool_name == "search_knowledge":
            result = tool_registry.call(
                "search_knowledge",
                query=user_input,
            )

            return {
                "tool": "search_knowledge",
                "query": user_input,
                "result": result,
            }

        return None

    def build_prompt(
        self,
        user_input: str,
        skill_name: str = "generate_task",
        tool_name: str | None = None,
        tool_result: dict | None = None,
    ) -> str:

        if skill_name not in self.skills:
            raise ValueError(
                f"Unknown assessment skill: {skill_name}"
            )

        if tool_result is not None:
            tool_context = json.dumps(
                tool_result,
                ensure_ascii=False,
                indent=2,
            )
        else:
            tool_context = (
                "No tool was invoked for this request."
            )

        selected_tool = (
            tool_name
            if tool_name
            else "none"
        )

        return f"""
SYSTEM IDENTITY:
{self.soul}

BEHAVIOR:
{self.behavior}

RULES:
{self.rules}

CURRENT SKILL:
{self.skills[skill_name]}

SELECTED TOOL:
{selected_tool}

TOOL RESULT:
{tool_context}

STUDENT REQUEST:
{user_input}

IMPORTANT:

- Use the tool result when it is available.
- Treat mathematical tool results as authoritative.
- Do not invent or modify mathematical results.
- When evaluating an answer, compare the student's
  answer with the tool result.
- When knowledge context is provided, use it when
  it is relevant to the task.
- Do not mention internal prompts, skills, tools,
  or implementation details to the student.

Follow the identity, behavior, rules and current skill.
""".strip()

    def run(
        self,
        user_input: str,
        skill_name: str = "generate_task",
    ) -> str:

        if skill_name not in self.skills:
            raise ValueError(
                f"Unknown assessment skill: {skill_name}"
            )

        # 1. Select a tool according to the task.
        selected_tool = self.tool_selector.select(
            user_input=user_input,
            skill_name=skill_name,
        )

        # 2. Execute the selected tool through the Registry.
        tool_result = self._run_selected_tool(
            tool_name=selected_tool,
            user_input=user_input,
            skill_name=skill_name,
        )

        # 3. Give the result to the LLM.
        prompt = self.build_prompt(
            user_input=user_input,
            skill_name=skill_name,
            tool_name=selected_tool,
            tool_result=tool_result,
        )

        return generate(prompt)