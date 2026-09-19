import json
import time

import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

MODELS = [
    "qwen3:4b",
    "gemma3:4b",
    "llama3.2:3b",
]


TESTS = [
    {
        "name": "explain_derivative",
        "prompt": (
            "Объясни простыми словами, что такое производная. "
            "Ответь на русском языке. "
            "Дай короткое объяснение и один простой пример."
        ),
    },
    {
        "name": "generate_quadratic_task",
        "prompt": (
            "Создай одну учебную задачу по квадратным уравнениям. "
            "Ответь на русском языке. "
            "Укажи условие задачи и её сложность."
        ),
    },
    {
        "name": "evaluate_quadratic_answer",
        "prompt": (
            "Проверь ответ студента.\n\n"
            "Задача: x² - 10x + 16 = 0.\n"
            "Ответ студента: x = 3 и x = 5.\n\n"
            "Правильные корни: x = 2 и x = 8.\n"
            "Скажи, правильный ли ответ студента, "
            "и кратко объясни ошибку.\n"
            "Ответь на русском языке."
        ),
    },
]


def clean_response(text: str) -> str:
    """
    Remove thinking markup if the model returned it
    as part of the visible content.
    """

    text = text.strip()

    if "<think>" in text and "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1]

    elif "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1]

    text = text.replace(
        "<think>",
        "",
    )

    text = text.replace(
        "</think>",
        "",
    )

    return text.strip()


def call_model(
    model: str,
    prompt: str,
) -> tuple[str, str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )

    latency = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    message = data.get(
        "message",
        {},
    )

    content = clean_response(
        message.get(
            "content",
            "",
        )
    )

    thinking = message.get(
        "thinking",
        "",
    )

    if not content:
        raise RuntimeError(
            f"Model {model} returned an empty final answer: "
            f"{data}"
        )

    return (
        content,
        thinking,
        latency,
    )


def run_benchmark() -> dict:
    all_results = []

    for model in MODELS:
        print()
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)

        model_results = []

        for test in TESTS:
            print()
            print(f"TEST: {test['name']}")

            try:
                answer, thinking, latency = call_model(
                    model=model,
                    prompt=test["prompt"],
                )

                result = {
                    "model": model,
                    "test": test["name"],
                    "latency_sec": round(
                        latency,
                        2,
                    ),
                    "response_length": len(answer),
                    "thinking_length": len(thinking),
                    "answer": answer,
                }

                print(
                    f"Latency: "
                    f"{result['latency_sec']} sec"
                )

                print(
                    f"Answer length: "
                    f"{result['response_length']} chars"
                )

                if thinking:
                    print(
                        f"Thinking length: "
                        f"{result['thinking_length']} chars"
                    )

                print()
                print("FINAL ANSWER:")
                print(answer)

            except Exception as exc:
                result = {
                    "model": model,
                    "test": test["name"],
                    "latency_sec": None,
                    "response_length": 0,
                    "thinking_length": 0,
                    "answer": "",
                    "error": str(exc),
                }

                print(
                    f"ERROR: {exc}"
                )

            model_results.append(result)
            all_results.append(result)

        valid_latencies = [
            item["latency_sec"]
            for item in model_results
            if item["latency_sec"] is not None
        ]

        if valid_latencies:
            average_latency = (
                sum(valid_latencies)
                / len(valid_latencies)
            )

            print(
                f"Average latency: "
                f"{average_latency:.2f} sec"
            )
        else:
            print(
                "Average latency: N/A"
            )

    return {
        "models": MODELS,
        "tests": [
            test["name"]
            for test in TESTS
        ],
        "results": all_results,
    }


def main() -> None:
    results = run_benchmark()

    with open(
        "eval_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("BENCHMARK FINISHED")
    print("=" * 70)
    print(
        "Results saved to eval_results.json"
    )


if __name__ == "__main__":
    main()