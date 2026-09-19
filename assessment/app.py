from flask import Flask, request, jsonify

from assessment.agent import AssessmentAgent


app = Flask(__name__)
agent = AssessmentAgent()


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json() or {}

    user_input = data.get(
        "input",
        ""
    ).strip()

    skill_name = data.get(
        "skill",
        "generate_task"
    )

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

    try:
        answer = agent.run(
            user_input=user_input,
            skill_name=skill_name,
            student_id=student_id,
            session_id=session_id,
        )

        return jsonify({
            "agent": "assessment",
            "response": answer,
            "skill": skill_name,
            "student_id": student_id,
            "session_id": session_id,
            "memory_used": True,
        })

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5002,
    )