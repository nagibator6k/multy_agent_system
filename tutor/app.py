from flask import Flask, request, jsonify
from tutor.agent import TutorAgent
from rag.rag import search

app = Flask(__name__)
agent = TutorAgent()


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json() or {}

    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({"error": "input is required"}), 400

    context = search(user_input)

    answer = agent.run(
        user_input=user_input,
        context=context,
    )

    return jsonify({
        "agent": "tutor",
        "response": answer,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)