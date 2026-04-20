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
    user_input = request.json["input"]

    with langfuse.start_as_current_observation(
        name="orchestrator",
        as_type="generation",
        input=user_input,
        model="router"
    ) as gen:

        agent = route(user_input)

        if agent == "tutor":
            res = requests.post(TUTOR_URL, json={"input": user_input})
        else:
            res = requests.post(ASSESS_URL, json={"input": user_input})

        output = res.json()["response"]

        gen.update(
            output=output,
            usage_details={
                "input": count_tokens(user_input),
                "output": count_tokens(output)
            },
            metadata={"agent": agent}
        )

    return jsonify({"response": output})

app.run(host="0.0.0.0", port=5000)