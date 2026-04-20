from flask import Flask, request, jsonify
import requests
from langfuse_config import langfuse
from shared.tokens import count_tokens

OLLAMA_URL = "http://ollama:11434/api/generate"

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    user_input = request.json["input"]

    prompt = f"""
Ты экзаменатор.

Создай задание или проверь ответ.

Запрос:
{user_input}
"""

    with langfuse.start_as_current_observation(
        name="assessment",
        as_type="generation",
        input=prompt,
        model="qwen2.5:3b"
    ) as gen:

        res = requests.post(
            OLLAMA_URL,
            json={"model": "qwen3:4b", "prompt": prompt}
        )

        output = res.json()["response"]

        gen.update(
            output=output,
            usage_details={
                "input": count_tokens(prompt),
                "output": count_tokens(output)
            },
            metadata={"agent": "assessment"}
        )

    return jsonify({"response": output})

app.run(host="0.0.0.0", port=5002)