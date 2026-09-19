import re


class ToolSelector:
    """
    Selects an appropriate tool based on the task.

    This is a deterministic selector for now.
    Later it can be replaced with an LLM-based selector
    without changing the Tool Registry.
    """

    QUADRATIC_PATTERN = re.compile(
        r"[+-]?\d*\.?\d*x(?:\^2|²)"
        r".*[+-].*x"
        r".*=\s*0",
        re.IGNORECASE,
    )

    KNOWLEDGE_KEYWORDS = (
        "объясни",
        "что такое",
        "расскажи",
        "определи",
        "понятие",
        "формула",
        "производн",
        "интеграл",
        "алгебр",
        "математ",
        "уравнен",
        "функци",
        "дискриминант",
    )

    def select(
        self,
        user_input: str,
        skill_name: str,
    ) -> str | None:
        text = user_input.lower().strip()

        # Mathematical tool for quadratic equations.
        if skill_name == "evaluate_answer":
            if self._contains_quadratic_equation(text):
                return "solve_quadratic"

        # Knowledge search for educational requests.
        if skill_name in {
            "generate_task",
            "give_feedback",
            "evaluate_answer",
        }:
            if any(
                keyword in text
                for keyword in self.KNOWLEDGE_KEYWORDS
            ):
                return "search_knowledge"

        return None

    def _contains_quadratic_equation(
        self,
        text: str,
    ) -> bool:
        normalized = (
            text
            .replace("²", "^2")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
        )

        normalized = re.sub(
            r"\s+",
            "",
            normalized,
        )

        return bool(
            self.QUADRATIC_PATTERN.search(normalized)
        )