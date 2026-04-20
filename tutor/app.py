from flask import Flask, request, jsonify
import requests
from rag import search
from langfuse_config import langfuse
from shared.tokens import count_tokens

OLLAMA_URL = "http://ollama:11434/api/generate"

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    user_input = request.json["input"]

    context = search(user_input)

    prompt = f"""
Ты преподаватель.

Контекст:
{context}

Вопрос:
{user_input}
"""

    with langfuse.start_as_current_observation(
        name="tutor",
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
            metadata={"agent": "tutor"}
        )

    return jsonify({"response": output})

app.run(host="0.0.0.0", port=5001)