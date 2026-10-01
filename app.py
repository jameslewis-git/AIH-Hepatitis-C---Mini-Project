"""
Hepatitis C Detection and Staging using Machine Learning
Flask web application (HTML + Bootstrap frontend, ML backend).
Run: python flask_app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify

MODEL_PATH = "model/hcv_model.pkl"

app = Flask(__name__)

STAGE_INFO = {
    "Blood Donor": {
        "color": "#5c8a68",
        "desc": "No indication of liver disease. Lab values are within a healthy donor range.",
    },
    "Suspect Blood Donor": {
        "color": "#c08a2e",
        "desc": "Borderline values. Not a confirmed donor profile — further testing recommended.",
    },
    "Hepatitis": {
        "color": "#c08a2e",
        "desc": "Pattern consistent with active Hepatitis C infection. Clinical follow-up advised.",
    },
    "Fibrosis": {
        "color": "#a34630",
        "desc": "Signs of liver fibrosis (scarring) consistent with disease progression.",
    },
    "Cirrhosis": {
        "color": "#6e2330",
        "desc": "Advanced liver scarring. Urgent specialist consultation strongly recommended.",
    },
}

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None and os.path.exists(MODEL_PATH):
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(bundle, inputs: dict):
    feature_cols = bundle["feature_cols"]
    row = pd.DataFrame([inputs])[feature_cols]
    row_imputed = bundle["imputer"].transform(row)
    row_scaled = bundle["scaler"].transform(row_imputed)
    proba = bundle["model"].predict_proba(row_scaled)[0]
    pred_idx = int(np.argmax(proba))
    label = bundle["class_labels"][pred_idx]
    return label, proba, bundle["class_labels"]


@app.route("/")
def home():
    bundle = get_bundle()
    accuracy = f"{bundle['accuracy']*100:.1f}%" if bundle else "N/A"
    model_name = bundle["model_name"] if bundle else "N/A"
    return render_template("home.html", accuracy=accuracy, model_name=model_name)


@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    bundle = get_bundle()
    if bundle is None:
        return render_template("predict.html", error="Model not found. Run train_model.py first.", form=request.form)

    if request.method == "GET":
        return render_template("predict.html", form={})

    try:
        inputs = {
            "Age": float(request.form["age"]),
            "Sex": 1 if request.form["sex"] == "Male" else 0,
            "ALB": float(request.form["alb"]),
            "ALP": float(request.form["alp"]),
            "ALT": float(request.form["alt"]),
            "AST": float(request.form["ast"]),
            "BIL": float(request.form["bil"]),
            "CHE": float(request.form["che"]),
            "CHOL": float(request.form["chol"]),
            "CREA": float(request.form["crea"]),
            "GGT": float(request.form["ggt"]),
            "PROT": float(request.form["prot"]),
        }
    except (KeyError, ValueError):
        return render_template("predict.html", error="Please fill all fields with valid numbers.", form=request.form)

    label, proba, class_labels = predict(bundle, inputs)
    confidence = round(float(np.max(proba)) * 100, 1)
    info = STAGE_INFO[label]

    probs = sorted(
        [{"stage": class_labels[i], "value": round(float(p) * 100, 1)} for i, p in enumerate(proba)],
        key=lambda x: x["value"],
        reverse=True,
    )

    return render_template(
        "result.html",
        label=label,
        confidence=confidence,
        color=info["color"],
        desc=info["desc"],
        probs=probs,
        inputs=request.form,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "Model not found. Run train_model.py first."}), 500

    data = request.get_json(force=True)
    try:
        inputs = {
            "Age": float(data["age"]),
            "Sex": 1 if data["sex"] == "Male" else 0,
            "ALB": float(data["alb"]),
            "ALP": float(data["alp"]),
            "ALT": float(data["alt"]),
            "AST": float(data["ast"]),
            "BIL": float(data["bil"]),
            "CHE": float(data["che"]),
            "CHOL": float(data["chol"]),
            "CREA": float(data["crea"]),
            "GGT": float(data["ggt"]),
            "PROT": float(data["prot"]),
        }
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Please fill all fields with valid numbers."}), 400

    label, proba, class_labels = predict(bundle, inputs)
    confidence = round(float(np.max(proba)) * 100, 1)
    info = STAGE_INFO[label]

    probs = sorted(
        [{"stage": class_labels[i], "value": round(float(p) * 100, 1)} for i, p in enumerate(proba)],
        key=lambda x: x["value"],
        reverse=True,
    )

    normal_ranges = {
        "ALB": (35, 50), "ALP": (40, 130), "ALT": (7, 56), "AST": (10, 40),
        "BIL": (3, 17), "CHE": (4.5, 11.5), "CHOL": (3.0, 5.2),
        "CREA": (60, 110), "GGT": (8, 61), "PROT": (64, 83),
    }
    flags = []
    for key, (lo, hi) in normal_ranges.items():
        val = inputs[key]
        if val < lo:
            flags.append({"marker": key, "value": val, "status": "low"})
        elif val > hi:
            flags.append({"marker": key, "value": val, "status": "high"})
        else:
            flags.append({"marker": key, "value": val, "status": "normal"})

    return jsonify({
        "label": label,
        "confidence": confidence,
        "color": info["color"],
        "desc": info["desc"],
        "probs": probs,
        "flags": flags,
    })


@app.route("/about")
def about():
    bundle = get_bundle()
    stages = [{"name": k, "color": v["color"], "desc": v["desc"]} for k, v in STAGE_INFO.items()]
    return render_template(
        "about.html",
        model_name=bundle["model_name"] if bundle else "N/A",
        accuracy=f"{bundle['accuracy']*100:.2f}%" if bundle else "N/A",
        features=bundle["feature_cols"] if bundle else [],
        stages=stages,
    )


if __name__ == "__main__":
    import socket

    port_env = os.environ.get("PORT")
    if port_env:
        port = int(port_env)
    else:
        # Check if 5000 is already in use (common conflict with macOS AirPlay receiver)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port = 5001 if s.connect_ex(("127.0.0.1", 5000)) == 0 else 5000

    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

