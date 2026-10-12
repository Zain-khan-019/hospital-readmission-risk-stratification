# Backend: Flask API
import json, joblib, pandas as pd
from flask import Flask, jsonify, render_template, request

NUM = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
       "number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
app = Flask(__name__)
models = joblib.load("model/models.joblib")
metrics = json.load(open("model/metrics.json"))
RISK = {"<30": "High", ">30": "Medium", "NO": "Low"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model = models[data.pop("model", "Random Forest")]
    if data["diag_1"] not in metrics["diag_top"]:
        data["diag_1"] = "Other"
    row = pd.DataFrame([data])
    for c in NUM:
        row[c] = pd.to_numeric(row[c])
    proba = model.predict_proba(row)[0]
    label = str(model.classes_[proba.argmax()])
    return jsonify({"label": label, "risk": RISK[label],
                    "probabilities": {str(c): float(p) for c, p in zip(model.classes_, proba)}})

if __name__ == "__main__":
    app.run(debug=True)
