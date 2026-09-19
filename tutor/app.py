from flask import Flask, request, jsonify
from tutor.agent import TutorAgent


app = Flask(__name__)
agent = TutorAgent()


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json() or {}

    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({
            "error": "input is required"
        }), 400

    answer = agent.run(user_input)

    return jsonify({
        "agent": "tutor",
        "response": answer,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
    )