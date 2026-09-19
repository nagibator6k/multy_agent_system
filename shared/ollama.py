import os
import re

import requests


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://ollama:11434",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemma3:4b",
)


def _clean_response(text: str) -> str:
    """
    Remove internal reasoning from the final response.

    Qwen3 can sometimes return its reasoning directly inside
    message.content without <think>...</think> markers.

    In our educational agents the final answer normally starts
    with one of the known output sections.
    """

    text = text.strip()

    # Remove explicit thinking blocks.
    if "<think>" in text and "</think>" in text:
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    if "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1]

    # Qwen CLI/API can also expose this marker.
    for marker in (
        "...done thinking...",
        "...done thinking.",
    ):
        if marker in text:
            text = text.split(
                marker,
                1,
            )[1]

    # Known final-answer headings for Tutor.
    tutor_markers = (
        "### Definition",
        "### Определение",
        "### Explanation",
        "### Объяснение",
        "### Example",
        "### Пример",
    )

    # Known final-answer headings for Assessment.
    assessment_markers = (
        "### Result",
        "### Результат",
        "### Task",
        "### Задача",
        "### Score",
        "### Оценка",
    )

    # Find the earliest final-answer marker.
    positions = []

    for marker in tutor_markers + assessment_markers:
        position = text.find(marker)

        if position >= 0:
            positions.append(position)

    if positions:
        first_position = min(positions)

        # Only cut text when there is actually a substantial
        # reasoning section before the final answer.
        if first_position > 0:
            text = text[first_position:]

    # Handle plain "Задача" output without markdown heading.
    lines = text.splitlines()

    if lines:
        for index, line in enumerate(lines):
            normalized = line.strip().lower()

            if normalized in {
                "задача",
                "task",
            } and index > 0:
                previous_text = "\n".join(
                    lines[:index]
                ).strip()

                # Avoid cutting on an accidental mention of
                # the word "задача" in normal final text.
                if len(previous_text) > 100:
                    text = "\n".join(
                        lines[index:]
                    )
                    break

    # Remove stray thinking tags.
    text = text.replace(
        "<think>",
        "",
    )

    text = text.replace(
        "</think>",
        "",
    )

    return text.strip()


def generate(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Generate a student-facing response through Ollama.

    Qwen3 is explicitly requested to use non-thinking mode.
    The response is additionally cleaned before returning it
    to the agent.
    """

    user_prompt = (
        "/no_think\n\n"
        + prompt
    )

    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,
        },
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    message = data.get(
        "message",
        {},
    )

    content = message.get(
        "content",
        "",
    )

    if not content:
        raise RuntimeError(
            f"Ollama returned an empty response: {data}"
        )

    return _clean_response(content)