import requests
import time
import json

URL = "http://127.0.0.1:5000/handle"

questions = [
    "What is algebra?",
    "Explain AI agent in simple terms",
    "Create a math task about derivatives",
    "What is supervised learning?"
]

results = []

for q in questions:
    start = time.time()

    response = requests.post(
        URL,
        json={"input": q}
    )

    end = time.time()

    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")
    else:
        answer = f"ERROR: {response.status_code}"

    results.append({
        "question": q,
        "answer": answer,
        "latency_sec": round(end - start, 2)
    })

# save results
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE. Results saved to eval_results.json")