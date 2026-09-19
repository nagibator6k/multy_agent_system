from flask import Flask, request, jsonify
import requests

from router import route
from langfuse_config import langfuse
from shared.tokens import count_tokens


app = Flask(__name__)

TUTOR_URL = "http://tutor:5001/run"
ASSESS_URL = "http://assessment:5002/run"


@app.route("/handle", methods=["POST"])
def handle():
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

    with langfuse.start_as_current_observation(
        name="orchestrator",
        as_type="generation",
        input=user_input,
        model="router",
    ) as gen:

        agent = route(user_input)

        payload = {
            "input": user_input,
            "student_id": student_id,
            "session_id": session_id,
        }

        if agent == "tutor":
            res = requests.post(
                TUTOR_URL,
                json=payload,
            )
        else:
            res = requests.post(
                ASSESS_URL,
                json=payload,
            )

        res.raise_for_status()

        output = res.json()["response"]

        gen.update(
            output=output,
            usage_details={
                "input": count_tokens(user_input),
                "output": count_tokens(output),
            },
            metadata={
                "agent": agent,
                "student_id": student_id,
                "session_id": session_id,
            },
        )

    return jsonify({
        "response": output,
        "agent": agent,
        "student_id": student_id,
        "session_id": session_id,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
    )