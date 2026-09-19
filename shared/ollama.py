import os

import requests


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://ollama:11434",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen3:4b",
)


def generate(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    payload = {
        "model": model or MODEL_NAME,
        "prompt": prompt,
        "stream": False,

        # Qwen3 supports explicit thinking control.
        # For our educational agents we use non-thinking
        # mode to avoid unnecessarily long generation.
        "think": False,

        "options": {
            "temperature": temperature,

            # Limit the generated answer length.
            # This is enough for our tutor/assessment
            # responses while preventing very long generations.
            "num_predict": 768,
        },
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    return data["response"]