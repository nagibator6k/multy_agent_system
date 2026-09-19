import json
import re

from shared.ollama import generate
from shared.skill_loader import load_skill
from tools.registry import tool_registry
from tools.selector import ToolSelector
from memory.store import MemoryStore


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
        self.memory = MemoryStore()

    @staticmethod
    def _detect_language(
        user_input: str,
    ) -> str:
        cyrillic_count = sum(
            1
            for char in user_input
            if "\u0400" <= char <= "\u04ff"
        )

        latin_count = sum(
            1
            for char in user_input
            if (
                "A" <= char <= "Z"
                or "a" <= char <= "z"
            )
        )

        if cyrillic_count > latin_count:
            return "Russian"

        return "English"

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

    @staticmethod
    def _extract_score(
        text: str,
    ) -> float | None:
        """
        Extract a score from common formats and normalize it
        to a percentage.

        Supported examples:

            80%
            8/10
            8 / 10
            8 из 10
            8 из 10 баллов
            0 баллов
            Оценка: 80%
            Score: 8/10
        """

        percentage_match = re.search(
            r"(?:оценка|score)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*%",
            text,
            re.IGNORECASE,
        )

        if percentage_match:
            score = float(
                percentage_match.group(1)
            )

            return max(
                0.0,
                min(100.0, score),
            )

        fraction_match = re.search(
            r"(?:оценка|score)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*/\s*"
            r"(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )

        if fraction_match:
            numerator = float(
                fraction_match.group(1)
            )

            denominator = float(
                fraction_match.group(2)
            )

            if denominator == 0:
                return None

            score = (
                numerator
                / denominator
                * 100
            )

            return max(
                0.0,
                min(100.0, score),
            )

        russian_fraction_match = re.search(
            r"(?:оценка)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s+из\s+"
            r"(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )

        if russian_fraction_match:
            numerator = float(
                russian_fraction_match.group(1)
            )

            denominator = float(
                russian_fraction_match.group(2)
            )

            if denominator == 0:
                return None

            score = (
                numerator
                / denominator
                * 100
            )

            return max(
                0.0,
                min(100.0, score),
            )

        points_match = re.search(
            r"(?:оценка)?\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"(?:балл(?:а|ов)?|points?)",
            text,
            re.IGNORECASE,
        )

        if points_match:
            points = float(
                points_match.group(1)
            )

            score = (
                points
                / 10.0
                * 100
            )

            return max(
                0.0,
                min(100.0, score),
            )

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
        memory_context: str = "",
    ) -> str:

        if skill_name not in self.skills:
            raise ValueError(
                f"Unknown assessment skill: {skill_name}"
            )

        language = self._detect_language(
            user_input
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

STUDENT LANGUAGE:
{language}

SELECTED TOOL:
{selected_tool}

TOOL RESULT:
{tool_context}

STUDENT MEMORY:
{memory_context if memory_context else "No stored memory is available."}

STUDENT REQUEST:
{user_input}

FINAL RESPONSE RULES:

1. Return ONLY the final answer for the student.
2. NEVER output internal reasoning or analysis.
3. NEVER describe your reasoning process.
4. NEVER write phrases such as:
   "We are given",
   "We need to",
   "Let's analyze",
   "But note",
   "Wait",
   "The system says",
   "The tool result says",
   or similar internal reasoning.
5. Do not mention prompts, system instructions,
   skills, tools, memory, databases, or implementation.
6. Answer entirely in the student's language.
7. If the student writes in Russian, use Russian
   headings and Russian explanations.
8. If the student writes in English, use English.
9. Generate exactly what the current skill requires.
10. Do not add unrelated explanations.

SPECIAL RULES FOR generate_task:

- Create ONE clear educational task.
- Do not reveal the internal reasoning used to create it.
- Do not discuss why the task was chosen.
- Do not discuss the student's internal memory.
- Do not reproduce previous mistakes unless the task
  itself needs them.
- Do not reveal an "expected answer" to the student
  unless the current skill explicitly requires it.
- Prefer a concise student-facing task.

SPECIAL RULES FOR evaluate_answer:

- Use the mathematical tool result as authoritative
  when it is available.
- Compare the student's answer with the tool result.
- Give an explicit score.
- Explain mistakes clearly.
- Give actionable feedback.

Follow the identity, behavior, rules and current skill.
""".strip()

    def run(
        self,
        user_input: str,
        skill_name: str = "generate_task",
        student_id: str = "demo-user",
        session_id: str = "default-session",
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

        # 2. Execute the selected tool.
        tool_result = self._run_selected_tool(
            tool_name=selected_tool,
            user_input=user_input,
            skill_name=skill_name,
        )

        # 3. Retrieve student memory.
        memory_context = self.memory.get_student_context(
            student_id=student_id,
            session_id=session_id,
        )

        # 4. Build the final prompt.
        prompt = self.build_prompt(
            user_input=user_input,
            skill_name=skill_name,
            tool_name=selected_tool,
            tool_result=tool_result,
            memory_context=memory_context,
        )

        # 5. Generate the final student-facing answer.
        answer = generate(prompt)

        # 6. Save short-term conversation memory.
        self.memory.save_message(
            student_id=student_id,
            session_id=session_id,
            role="user",
            content=user_input,
            agent="assessment",
        )

        self.memory.save_message(
            student_id=student_id,
            session_id=session_id,
            role="assistant",
            content=answer,
            agent="assessment",
        )

        # 7. Extract and save assessment score.
        score = self._extract_score(answer)

        self.memory.save_assessment(
            student_id=student_id,
            session_id=session_id,
            skill=skill_name,
            request=user_input,
            response=answer,
            score=score,
        )

        # 8. Save learning progress and mistakes.
        if score is not None:
            topic = (
                "quadratic_equation"
                if selected_tool == "solve_quadratic"
                else skill_name
            )

            normalized_score = max(
                0.0,
                min(100.0, score),
            )

            self.memory.save_progress(
                student_id=student_id,
                topic=topic,
                mastery_score=normalized_score,
            )

            if normalized_score < 100.0:
                self.memory.save_mistake(
                    student_id=student_id,
                    topic=topic,
                    description=(
                        f"Assessment score: "
                        f"{normalized_score:.1f}%. "
                        f"Request: {user_input}"
                    ),
                )

        return answer