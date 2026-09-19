import json
import subprocess
import time


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
            "Дай короткое объяснение и один простой пример. "
            "Не показывай внутреннее рассуждение."
        ),
    },
    {
        "name": "generate_quadratic_task",
        "prompt": (
            "Создай одну учебную задачу по квадратным уравнениям. "
            "Ответь на русском языке. "
            "Укажи условие задачи и сложность. "
            "Не показывай внутреннее рассуждение."
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
            "Ответь на русском языке. "
            "Не показывай внутреннее рассуждение."
        ),
    },
]


def clean_response(text: str) -> str:
    """
    Extract the final answer from Ollama CLI output.

    Qwen3 may print a visible thinking block in CLI mode:

        Thinking...
        ...
        ...done thinking.

        FINAL ANSWER

    In that case we keep only the text after
    "...done thinking.".
    """

    text = text.strip()

    thinking_markers = [
        "...done thinking...",
        "...done thinking.",
    ]

    for marker in thinking_markers:
        if marker in text:
            text = text.split(
                marker,
                1,
            )[1]

    if "</think>" in text:
        text = text.split(
            "</think>",
            1,
        )[1]

    if "<think>" in text:
        text = text.replace(
            "<think>",
            "",
        )

    if "</think>" in text:
        text = text.replace(
            "</think>",
            "",
        )

    return text.strip()


def run_model(
    model: str,
    prompt: str,
) -> tuple[str, float]:
    command = [
        "docker",
        "exec",
        "ollama",
        "ollama",
        "run",
        model,
        prompt,
    ]

    start = time.perf_counter()

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )

    latency = time.perf_counter() - start

    if process.returncode != 0:
        error = (
            process.stderr.strip()
            or process.stdout.strip()
            or f"Exit code: {process.returncode}"
        )

        raise RuntimeError(error)

    output = process.stdout.strip()

    if not output:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return (
        clean_response(output),
        latency,
    )


def run_benchmark() -> dict:
    results = []

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
                answer, latency = run_model(
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
                    "response_length": len(
                        answer
                    ),
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

                print()
                print("FINAL ANSWER:")
                print(answer)

            except Exception as exc:
                result = {
                    "model": model,
                    "test": test["name"],
                    "latency_sec": None,
                    "response_length": 0,
                    "answer": "",
                    "error": str(exc),
                }

                print(
                    f"ERROR: {exc}"
                )

            model_results.append(result)
            results.append(result)

        latencies = [
            item["latency_sec"]
            for item in model_results
            if item["latency_sec"] is not None
        ]

        if latencies:
            average_latency = (
                sum(latencies)
                / len(latencies)
            )

            print()
            print(
                f"Average latency: "
                f"{average_latency:.2f} sec"
            )
        else:
            print()
            print(
                "Average latency: N/A"
            )

    return {
        "models": MODELS,
        "tests": [
            test["name"]
            for test in TESTS
        ],
        "results": results,
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