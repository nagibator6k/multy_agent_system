from flask import Flask, request, jsonify

from tutor.agent import TutorAgent


app = Flask(__name__)
agent = TutorAgent()


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json() or {}

    user_input = data.get(
        "input",
        ""
    ).strip()

    student_id = data.get(
        "student_id",
        "demo-user"
    )

    session_id = data.get(
        "session_id",
        "default-session"
    )

    if not user_input:
        return jsonify({
            "error": "input is required"
        }), 400

    answer = agent.run(
        user_input=user_input,
        student_id=student_id,
        session_id=session_id,
    )

    return jsonify({
        "agent": "tutor",
        "response": answer,
        "student_id": student_id,
        "session_id": session_id,
        "memory_used": True,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
    )