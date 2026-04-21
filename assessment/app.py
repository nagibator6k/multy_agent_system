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

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen3:4b",
                "prompt": prompt,
                "stream": False
            })

        data = res.json()
        output = data.get("response","")

    except Exception as e:
        output = f"OLLAMA ERROR: {str(e)}"

    return jsonify({"response": output})

app.run(host="0.0.0.0", port=5002)