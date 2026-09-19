import re


TUTOR_PATTERNS = [
    # Русский
    r"\bобъясни\b",
    r"\bобъяснить\b",
    r"\bрасскажи\b",
    r"\bрассказать\b",
    r"\bчто такое\b",
    r"\bкак работает\b",
    r"\bкак устроен\b",
    r"\bпочему\b",
    r"\bзачем\b",
    r"\bпомоги понять\b",
    r"\bне понимаю\b",
    r"\bобъяснение\b",
    r"\bкакие ошибки\b",
    r"\bмои ошибки\b",
    r"\bошибки я\b",
    r"\bчто я изучаю\b",
    r"\bчто я сейчас изучаю\b",

    # English
    r"\bexplain\b",
    r"\bwhat is\b",
    r"\bhow does\b",
    r"\bhow do\b",
    r"\bwhy\b",
    r"\bhelp me understand\b",
    r"\bwhat am i studying\b",
    r"\bmy mistakes\b",
    r"\bwhat mistakes\b",
]


ASSESSMENT_PATTERNS = [
    # Русский
    r"\bсоздай задачу\b",
    r"\bсоздай задание\b",
    r"\bдай задачу\b",
    r"\bдай задание\b",
    r"\bсгенерируй задачу\b",
    r"\bсгенерируй задание\b",
    r"\bсделай тест\b",
    r"\bсоздай тест\b",
    r"\bпроверь мой ответ\b",
    r"\bпроверь ответ\b",
    r"\bоцени мой ответ\b",
    r"\bоцени ответ\b",
    r"\bпроверь решение\b",
    r"\bоцени решение\b",
    r"\bдай обратную связь\b",
    r"\bпроверь\b",
    r"\bоцени\b",
    r"\bрешение\b",

    # English
    r"\bcreate a task\b",
    r"\bgenerate a task\b",
    r"\bcreate an exercise\b",
    r"\bgenerate an exercise\b",
    r"\bcreate a test\b",
    r"\bgenerate a test\b",
    r"\bcheck my answer\b",
    r"\bcheck the answer\b",
    r"\bevaluate my answer\b",
    r"\bevaluate the answer\b",
    r"\bcheck my solution\b",
    r"\bevaluate my solution\b",
    r"\bgive feedback\b",
]


def _normalize(text: str) -> str:
    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def _matches(
    text: str,
    patterns: list[str],
) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def route(text: str) -> str:
    """
    Route a student request to the appropriate agent.

    Returns:
        "tutor"      -> conceptual explanation,
                       learning context and memory.
        "assessment" -> task generation,
                       answer evaluation and feedback.

    Assessment patterns are checked first because requests
    such as "проверь решение" may also contain words that
    look like ordinary educational questions.
    """

    normalized = _normalize(text)

    if not normalized:
        return "assessment"

    # Evaluation/generation tasks have priority.
    if _matches(
        normalized,
        ASSESSMENT_PATTERNS,
    ):
        return "assessment"

    # Explanations and memory-related learning questions.
    if _matches(
        normalized,
        TUTOR_PATTERNS,
    ):
        return "tutor"

    # Default behavior remains assessment so that
    # previously supported requests continue to work.
    return "assessment"