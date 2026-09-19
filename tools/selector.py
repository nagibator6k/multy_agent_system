import re


class ToolSelector:
    """
    Selects an appropriate tool based on the task.

    The selector is deterministic and keeps tool choice
    outside the LLM. This makes tool usage predictable.
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

        if not text:
            return None

        # Mathematical tool is used only when evaluating
        # an explicit quadratic equation.
        if skill_name == "evaluate_answer":
            if self._contains_quadratic_equation(text):
                return "solve_quadratic"

        # Knowledge search is useful for feedback/evaluation
        # when the request explicitly refers to educational
        # knowledge.
        #
        # Do NOT use search_knowledge for generate_task by
        # default. A generic task-generation request does not
        # require a knowledge-base lookup.
        if skill_name in {
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
            self.QUADRATIC_PATTERN.search(
                normalized
            )
        )