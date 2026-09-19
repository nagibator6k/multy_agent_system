import os
import re

import requests


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://ollama:11434",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen3:4b",
)


def _clean_response(text: str) -> str:
    """
    Remove model reasoning from the final response.

    Qwen3 may sometimes return internal reasoning even when
    thinking is disabled. Depending on the Ollama/model version,
    the reasoning can appear inside <think>...</think> or before
    a closing </think> tag.

    The student should receive only the final answer.
    """

    text = text.strip()

    # Case 1:
    # <think>internal reasoning</think>
    # final answer
    if "<think>" in text and "</think>" in text:
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Case 2:
    # internal reasoning
    # </think>
    # final answer
    #
    # This handles models that omit the opening <think> tag.
    if "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1]

    # Remove a possible stray opening tag.
    text = text.replace(
        "<think>",
        "",
    )

    # Remove a possible stray closing tag.
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
    payload = {
        "model": model or MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
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

    message = data.get("message")

    if not message:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
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