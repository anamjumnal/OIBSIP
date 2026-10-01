"""Flask web app: python app.py  ->  http://127.0.0.1:5000"""
import os, joblib
from flask import Flask, render_template, request, jsonify
from utils import clean_text

app = Flask(__name__)
PATH = "model/spam_model.joblib"
if not os.path.exists(PATH):
    import train_model  # trains on first run
bundle = joblib.load(PATH)
model = bundle["model"]

@app.route("/")
def index():
    return render_template("index.html", model_name=bundle["name"])

@app.route("/predict", methods=["POST"])
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please enter some text."), 400
    X = [clean_text(text)]
    pred = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        conf = float(model.predict_proba(X)[0][pred])
    else:  # LinearSVC: squash decision score into a pseudo-confidence
        import math
        s = float(model.decision_function(X)[0]); conf = 1 / (1 + math.exp(-abs(s) * 2))
    return jsonify(label="spam" if pred else "ham", confidence=round(conf * 100, 1))

if __name__ == "__main__":
    app.run(debug=True)
