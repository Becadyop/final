from flask import Flask, request, jsonify
from .engine import EchoFeelingEngine

app = Flask(__name__)

engine = EchoFeelingEngine()

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json

    text = data.get("text")
    emoji = data.get("emoji")
    sticker = data.get("sticker")

    result = engine.analyze(text, emoji, sticker)

    return jsonify(result)

def start_server():
    app.run(host="0.0.0.0", port=5050)