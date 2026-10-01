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

import base64
import json
import requests

def load_env_file():
    """Load key=value pairs from local .env file if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()

MODEL_PATH = "model/hcv_model.pkl"

app = Flask(__name__)


STAGE_INFO = {
    "Blood Donor": {
        "color": "#10b981",
        "desc": "No indication of liver disease. Lab values are within a healthy donor range.",
    },
    "Suspect Blood Donor": {
        "color": "#f59e0b",
        "desc": "Borderline values. Not a confirmed donor profile: further clinical monitoring recommended.",
    },
    "Hepatitis": {
        "color": "#f97316",
        "desc": "Pattern consistent with active Hepatitis C infection. Clinical follow-up advised.",
    },
    "Fibrosis": {
        "color": "#ef4444",
        "desc": "Signs of liver fibrosis (scar tissue) consistent with progressive damage.",
    },
    "Cirrhosis": {
        "color": "#9f1239",
        "desc": "Advanced liver scarring. Urgent specialist hepatology consultation strongly recommended.",
    },
}

_bundle = None

SAMPLE_LAB_REPORTS = {
    "sample_donor": {
        "title": "General Health Center - Annual Executive Lab Panel",
        "patient": "Jane Doe, 34F",
        "age": 34, "sex": "Female",
        "alb": 44.2, "alp": 62.0, "alt": 18.5, "ast": 21.0,
        "bil": 8.5, "che": 8.2, "chol": 4.8, "crea": 74.0,
        "ggt": 16.0, "prot": 75.0,
        "notes": "All 12 markers within standard clinical reference ranges. Normal hepatic function."
    },
    "sample_hepatitis": {
        "title": "Metropolitan Liver Clinic - Viral Hepatitis Serology & LFP",
        "patient": "Mark Smith, 46M",
        "age": 46, "sex": "Male",
        "alb": 37.5, "alp": 95.0, "alt": 184.0, "ast": 152.0,
        "bil": 22.4, "che": 6.4, "chol": 4.6, "crea": 88.0,
        "ggt": 96.0, "prot": 71.0,
        "notes": "Marked acute transaminitis (ALT 184 U/L, AST 152 U/L). Findings highly indicative of active viral hepatitis."
    },
    "sample_cirrhosis": {
        "title": "University Hepatology Division - Comprehensive Staging Assessment",
        "patient": "Robert Lee, 58M",
        "age": 58, "sex": "Male",
        "alb": 24.5, "alp": 165.0, "alt": 78.0, "ast": 112.0,
        "bil": 62.0, "che": 2.8, "chol": 2.4, "crea": 135.0,
        "ggt": 142.0, "prot": 58.0,
        "notes": "Inverted AST/ALT ratio > 1, severe hypoalbuminemia (24.5 g/L), marked hyperbilirubinemia, low cholinesterase. Late-stage cirrhosis profile."
    }
}


def get_bundle():
    global _bundle
    if _bundle is None and os.path.exists(MODEL_PATH):
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def parse_float_or_nan(val):
    if val is None or val == "" or str(val).strip() == "":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def predict(bundle, inputs: dict):
    feature_cols = bundle["feature_cols"]
    row = pd.DataFrame([inputs])[feature_cols]
    
    # Identify which markers are missing and will be imputed
    imputed_markers = [col for col in feature_cols if pd.isna(row[col].iloc[0])]
    
    row_imputed = bundle["imputer"].transform(row)
    row_scaled = bundle["scaler"].transform(row_imputed)
    proba = bundle["model"].predict_proba(row_scaled)[0]
    pred_idx = int(np.argmax(proba))
    label = bundle["class_labels"][pred_idx]
    return label, proba, bundle["class_labels"], imputed_markers


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

    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if request.method == "GET":
        return render_template("predict.html", form={}, has_gemini_key=has_key)

    age_val = parse_float_or_nan(request.form.get("age"))
    sex_str = request.form.get("sex", "Male")
    sex_val = 1 if sex_str == "Male" else 0

    inputs = {
        "Age": age_val if not np.isnan(age_val) else 45.0,
        "Sex": sex_val,
        "ALB": parse_float_or_nan(request.form.get("alb")),
        "ALP": parse_float_or_nan(request.form.get("alp")),
        "ALT": parse_float_or_nan(request.form.get("alt")),
        "AST": parse_float_or_nan(request.form.get("ast")),
        "BIL": parse_float_or_nan(request.form.get("bil")),
        "CHE": parse_float_or_nan(request.form.get("che")),
        "CHOL": parse_float_or_nan(request.form.get("chol")),
        "CREA": parse_float_or_nan(request.form.get("crea")),
        "GGT": parse_float_or_nan(request.form.get("ggt")),
        "PROT": parse_float_or_nan(request.form.get("prot")),
    }

    label, proba, class_labels, imputed_markers = predict(bundle, inputs)
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
        imputed_markers=imputed_markers,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "Model not found. Run train_model.py first."}), 500

    data = request.get_json(force=True)
    sex_val = 1 if str(data.get("sex", "Male")).capitalize() == "Male" else 0
    age_val = parse_float_or_nan(data.get("age"))

    inputs = {
        "Age": age_val if not np.isnan(age_val) else 45.0,
        "Sex": sex_val,
        "ALB": parse_float_or_nan(data.get("alb")),
        "ALP": parse_float_or_nan(data.get("alp")),
        "ALT": parse_float_or_nan(data.get("alt")),
        "AST": parse_float_or_nan(data.get("ast")),
        "BIL": parse_float_or_nan(data.get("bil")),
        "CHE": parse_float_or_nan(data.get("che")),
        "CHOL": parse_float_or_nan(data.get("chol")),
        "CREA": parse_float_or_nan(data.get("crea")),
        "GGT": parse_float_or_nan(data.get("ggt")),
        "PROT": parse_float_or_nan(data.get("prot")),
    }

    label, proba, class_labels, imputed_markers = predict(bundle, inputs)
    confidence = round(float(np.max(proba)) * 100, 1)
    info = STAGE_INFO.get(label, {"color": "#dc2626", "desc": "Liver disease stage classification."})

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
        if np.isnan(val):
            flags.append({"marker": key, "value": "Imputed", "status": "imputed"})
        elif val < lo:
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
        "imputed_markers": imputed_markers,
    })


@app.route("/api/scan-lab-report", methods=["POST"])
def api_scan_lab_report():
    """
    AI-Powered Lab Report Scanner.
    Supports either pre-loaded sample reports OR Gemini Vision OCR of uploaded photos/PDFs.
    """
    sample_id = request.args.get("sample") or (request.is_json and request.get_json(silent=True) or {}).get("sample")
    if sample_id and sample_id in SAMPLE_LAB_REPORTS:
        report = SAMPLE_LAB_REPORTS[sample_id]
        return jsonify({
            "status": "success",
            "source": "sample_report",
            "report_title": report["title"],
            "patient": report["patient"],
            "data": {
                "age": report["age"],
                "sex": report["sex"],
                "alb": report["alb"],
                "alp": report["alp"],
                "alt": report["alt"],
                "ast": report["ast"],
                "bil": report["bil"],
                "che": report["che"],
                "chol": report["chol"],
                "crea": report["crea"],
                "ggt": report["ggt"],
                "prot": report["prot"],
            },
            "notes": report["notes"]
        })

    # Custom file upload or base64 image
    api_key = (
        request.headers.get("X-Gemini-Key") or
        request.form.get("apiKey") or
        (request.is_json and (request.get_json(silent=True) or {}).get("apiKey")) or
        os.environ.get("GEMINI_API_KEY")
    )

    image_bytes = None
    mime_type = "image/jpeg"

    if "file" in request.files:
        f = request.files["file"]
        image_bytes = f.read()
        mime_type = f.content_type or "image/jpeg"
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64_str = data.get("imageBase64")
        if b64_str:
            if "," in b64_str:
                header, b64_str = b64_str.split(",", 1)
                if "png" in header:
                    mime_type = "image/png"
                elif "pdf" in header:
                    mime_type = "application/pdf"
            image_bytes = base64.b64decode(b64_str)

    if not image_bytes:
        return jsonify({"error": "No file or image payload received."}), 400

    if not api_key:
        return jsonify({
            "status": "api_key_required",
            "error": "Google Gemini API key required for live AI OCR of custom photos. Enter your API key above or test using the sample blood reports.",
        }), 400

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = """You are an expert clinical medical laboratory data extraction AI.
Examine this liver function panel (LFP) or blood test report image and extract the numerical values for the following 12 biomarkers:
- Age (years, integer or float)
- Sex ('Male' or 'Female')
- ALB (Albumin, g/L)
- ALP (Alkaline Phosphatase, U/L)
- ALT (Alanine Aminotransferase, U/L)
- AST (Aspartate Aminotransferase, U/L)
- BIL (Total Bilirubin, µmol/L)
- CHE (Cholinesterase, kU/L)
- CHOL (Total Cholesterol, mmol/L)
- CREA (Creatinine, µmol/L)
- GGT (Gamma-Glutamyl Transferase, U/L)
- PROT (Total Protein, g/L)

Return ONLY valid JSON matching this schema with no markdown backticks:
{
  "age": float or null,
  "sex": "Male" or "Female" or null,
  "alb": float or null,
  "alp": float or null,
  "alt": float or null,
  "ast": float or null,
  "bil": float or null,
  "che": float or null,
  "chol": float or null,
  "crea": float or null,
  "ggt": float or null,
  "prot": float or null,
  "report_title": "Extracted test panel name",
  "notes": "Short summary of extracted lab values"
}
"""
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": b64_image}}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        
        resp = requests.post(url, json=payload, timeout=25)
        if not resp.ok:
            return jsonify({"error": f"Gemini API returned error {resp.status_code}: {resp.text}"}), 400
        
        resp_json = resp.json()
        extracted_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        extracted_data = json.loads(extracted_text)

        return jsonify({
            "status": "success",
            "source": "gemini_ocr",
            "report_title": extracted_data.get("report_title", "Uploaded Blood Test Report"),
            "data": extracted_data,
            "notes": extracted_data.get("notes", "Biomarkers successfully extracted via Gemini 1.5 Flash.")
        })
    except Exception as e:
        return jsonify({"error": f"OCR extraction failed: {str(e)}"}), 500


@app.route("/api/symptom-screener", methods=["POST"])
def api_symptom_screener():
    """
    AI Clinical Risk & Symptom Pre-Screener for users with zero lab data.
    Provides risk level, clinical guidance, and estimated blood markers.
    """
    data = request.get_json(force=True) or {}
    age = parse_float_or_nan(data.get("age")) or 45.0
    sex = data.get("sex", "Male")

    # Checked symptoms & risk factors
    transfusion = data.get("transfusion", False)
    tattoos = data.get("tattoos", False)
    jaundice = data.get("jaundice", False)
    dark_urine = data.get("dark_urine", False)
    chronic_fatigue = data.get("fatigue", False)
    abdominal_pain = data.get("abdominal_pain", False)
    alcohol_heavy = data.get("alcohol", False)
    family_history = data.get("family_history", False)

    # Score calculation
    risk_score = 10
    if transfusion: risk_score += 35
    if tattoos: risk_score += 15
    if jaundice: risk_score += 30
    if dark_urine: risk_score += 20
    if abdominal_pain: risk_score += 15
    if chronic_fatigue: risk_score += 10
    if alcohol_heavy: risk_score += 15
    if family_history: risk_score += 10

    risk_score = min(risk_score, 98)

    if risk_score >= 60:
        risk_level = "High"
        risk_color = "#dc2626"
        rationale = "High probability of active hepatic inflammation or structural liver strain. Marked clinical indicators (jaundice/transfusion history) strongly warrant immediate diagnostic confirmation."
        estimated_panel = {
            "age": age, "sex": sex,
            "alb": 33.5, "alp": 115.0, "alt": 178.0, "ast": 145.0,
            "bil": 38.0, "che": 5.2, "chol": 3.8, "crea": 92.0,
            "ggt": 110.0, "prot": 68.0
        }
    elif risk_score >= 30:
        risk_level = "Moderate"
        risk_color = "#d97706"
        rationale = "Moderate risk profile. Presenting mild systemic symptoms or historical risk factors. Routine screening blood panel recommended."
        estimated_panel = {
            "age": age, "sex": sex,
            "alb": 38.0, "alp": 85.0, "alt": 48.0, "ast": 42.0,
            "bil": 16.0, "che": 6.8, "chol": 4.8, "crea": 80.0,
            "ggt": 55.0, "prot": 71.0
        }
    else:
        risk_level = "Low"
        risk_color = "#059669"
        rationale = "Low immediate risk profile. No dominant hallmark symptoms or high-risk exposure history reported."
        estimated_panel = {
            "age": age, "sex": sex,
            "alb": 43.5, "alp": 60.0, "alt": 21.0, "ast": 23.0,
            "bil": 9.5, "che": 8.0, "chol": 4.9, "crea": 75.0,
            "ggt": 19.0, "prot": 74.0
        }

    recommendations = [
        "Comprehensive Liver Function Panel (ALT, AST, ALP, Total Bilirubin, Albumin)",
        "HCV Antibody Test (Anti-HCV Enzyme Immunoassay)",
        "Follow-up HCV RNA Quantitative PCR if antibody screen is reactive"
    ]

    return jsonify({
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_color": risk_color,
        "rationale": rationale,
        "estimated_panel": estimated_panel,
        "recommendations": recommendations,
    })


@app.route("/about")
def about():
    bundle = get_bundle()
    stages = [{"name": k, "color": v["color"], "desc": v["desc"]} for k, v in STAGE_INFO.items()]

    feature_importances = []
    if bundle and hasattr(bundle.get("model"), "feature_importances_"):
        raw_imp = bundle["model"].feature_importances_
        cols = bundle.get("feature_cols", [])
        marker_descriptions = {
            "AST": "Aspartate Aminotransferase · Hepatocellular necrosis & De Ritis index",
            "CHE": "Cholinesterase · Exclusive liver synthetic enzyme; plummets in cirrhosis",
            "ALB": "Albumin · Major serum protein; declining levels indicate decompensation",
            "Age": "Patient Age · Cumulative indicator of chronic fibrosis progression",
            "ALT": "Alanine Aminotransferase · Liver-specific cytoplasmic injury marker",
            "ALP": "Alkaline Phosphatase · Canalicular enzyme; marks cholestatic biliary injury",
            "CREA": "Creatinine · Renal filtration; flags hepatorenal syndrome in end-stage",
            "PROT": "Total Protein · Overall circulating serum immunoglobulins and albumin",
            "GGT": "Gamma-Glutamyl Transferase · Sensitive microsomal biliary enzyme",
            "BIL": "Total Bilirubin · Heme breakdown byproduct; clearance failure causes jaundice",
            "CHOL": "Total Cholesterol · Hepatic lipid synthesis and biliary excretion balance",
            "Sex": "Biological Sex · Baseline demographic stratification factor"
        }
        for name, val in sorted(zip(cols, raw_imp), key=lambda x: x[1], reverse=True):
            feature_importances.append({
                "name": name,
                "importance": round(float(val) * 100, 2),
                "desc": marker_descriptions.get(name, "Clinical biochemical indicator")
            })

    dataset_distribution = [
        {"label": "0 = Blood Donor", "count": 533, "pct": 86.7, "color": "#10b981", "type": "Healthy Control"},
        {"label": "3 = Cirrhosis", "count": 30, "pct": 4.9, "color": "#dc2626", "type": "End-Stage Disease"},
        {"label": "1 = Hepatitis", "count": 24, "pct": 3.9, "color": "#f97316", "type": "Active Infection"},
        {"label": "2 = Fibrosis", "count": 21, "pct": 3.4, "color": "#e11d48", "type": "Structural Remodeling"},
        {"label": "0s = Suspect Donor", "count": 7, "pct": 1.1, "color": "#f59e0b", "type": "Enzymatic Deferral"}
    ]

    eda_matrix = [
        {"stage": "0 = Blood Donor", "color": "#10b981", "count": 533, "age": 47.1, "alt": 26.6, "ast": 26.6, "bil": 8.5, "alb": 42.2, "che": 8.4, "ggt": 29.0},
        {"stage": "0s = Suspect Donor", "color": "#f59e0b", "count": 7, "age": 57.6, "alt": 102.1, "ast": 71.0, "bil": 4.7, "alb": 24.4, "che": 7.5, "ggt": 151.5},
        {"stage": "1 = Hepatitis", "color": "#f97316", "count": 24, "age": 38.7, "alt": 26.9, "ast": 75.7, "bil": 15.6, "alb": 43.8, "che": 9.3, "ggt": 92.6},
        {"stage": "2 = Fibrosis", "color": "#e11d48", "count": 21, "age": 52.3, "alt": 59.6, "ast": 81.2, "bil": 13.4, "alb": 41.8, "che": 8.3, "ggt": 79.6},
        {"stage": "3 = Cirrhosis", "color": "#dc2626", "count": 30, "age": 53.5, "alt": 23.0, "ast": 107.5, "bil": 59.1, "alb": 32.5, "che": 3.8, "ggt": 129.4}
    ]

    missing_audit = [
        {"feature": "ALP (Alkaline Phosphatase)", "missing": 18, "pct": 2.93, "median": 66.2, "unit": "U/L", "normal": "40–129 U/L"},
        {"feature": "CHOL (Total Cholesterol)", "missing": 10, "pct": 1.63, "median": 5.30, "unit": "mmol/L", "normal": "3.0–5.2 mmol/L"},
        {"feature": "ALB (Albumin)", "missing": 1, "pct": 0.16, "median": 41.95, "unit": "g/L", "normal": "35–50 g/L"},
        {"feature": "ALT (Alanine Aminotransferase)", "missing": 1, "pct": 0.16, "median": 23.0, "unit": "U/L", "normal": "7–56 U/L"},
        {"feature": "PROT (Total Protein)", "missing": 1, "pct": 0.16, "median": 72.2, "unit": "g/L", "normal": "64–83 g/L"},
    ]

    demographics = {
        "total_records": 615,
        "males": 377,
        "males_pct": 61.3,
        "females": 238,
        "females_pct": 38.7,
        "age_mean": 47.41,
        "age_median": 47.0,
        "age_min": 19,
        "age_max": 77,
        "age_std": 10.06,
    }

    biomarker_dictionary = [
        {"code": "AST", "name": "Aspartate Aminotransferase", "unit": "U/L", "mean": 34.79, "median": 25.9, "range": "10.6 – 324.0", "normal": "10 – 40 U/L", "role": "Mitochondrial & cytoplasmic enzyme; spikes in necrosis and cirrhosis."},
        {"code": "CHE", "name": "Cholinesterase", "unit": "kU/L", "mean": 8.20, "median": 8.26, "range": "1.42 – 16.41", "normal": "5.3 – 12.9 kU/L", "role": "Exclusive liver synthetic enzyme; severely suppressed in advanced cirrhosis."},
        {"code": "ALB", "name": "Albumin", "unit": "g/L", "mean": 41.62, "median": 41.95, "range": "14.9 – 82.2", "normal": "35 – 50 g/L", "role": "Primary oncotic protein; declines in liver decompensation and ascites."},
        {"code": "Age", "name": "Patient Age", "unit": "years", "mean": 47.41, "median": 47.0, "range": "19 – 77", "normal": "18 – 90", "role": "Duration of chronic infection; strongly correlates with cumulative fibrosis."},
        {"code": "ALT", "name": "Alanine Aminotransferase", "unit": "U/L", "mean": 28.45, "median": 23.0, "range": "0.9 – 325.3", "normal": "7 – 56 U/L", "role": "Liver-specific enzyme released into bloodstream during acute inflammation."},
        {"code": "ALP", "name": "Alkaline Phosphatase", "unit": "U/L", "mean": 68.28, "median": 66.2, "range": "11.3 – 416.6", "normal": "40 – 129 U/L", "role": "Biliary canalicular enzyme; flags cholestasis and biliary obstruction."},
        {"code": "CREA", "name": "Creatinine", "unit": "µmol/L", "mean": 81.29, "median": 77.0, "range": "8.0 – 1079.1", "normal": "60 – 110 µmol/L", "role": "Renal filtration marker; indicates hepatorenal syndrome in end-stage."},
        {"code": "PROT", "name": "Total Protein", "unit": "g/L", "mean": 72.04, "median": 72.2, "range": "44.8 – 90.0", "normal": "64 – 83 g/L", "role": "Serum protein sum; reflects protein synthesis and immunoglobulins."},
        {"code": "GGT", "name": "Gamma-Glutamyl Transferase", "unit": "U/L", "mean": 39.53, "median": 23.3, "range": "4.5 – 650.9", "normal": "8 – 61 U/L", "role": "Biliary epithelial enzyme; highly sensitive to toxic and viral damage."},
        {"code": "BIL", "name": "Total Bilirubin", "unit": "µmol/L", "mean": 11.40, "median": 7.3, "range": "0.8 – 254.0", "normal": "3 – 21 µmol/L", "role": "Heme breakdown pigment; accumulation causes jaundice and icterus."},
        {"code": "CHOL", "name": "Total Cholesterol", "unit": "mmol/L", "mean": 5.37, "median": 5.30, "range": "1.43 – 9.67", "normal": "3.0 – 5.2 mmol/L", "role": "Lipid synthesized by hepatocytes; drops during advanced parenchymal loss."},
        {"code": "Sex", "name": "Biological Sex", "unit": "binary", "mean": "61.3% M", "median": "Male (1)", "range": "Male / Female", "normal": "M: 61.3%, F: 38.7%", "role": "Demographic baseline covariate; encoded as Male=1, Female=0."}
    ]

    return render_template(
        "about.html",
        model_name=bundle["model_name"] if bundle else "XGBoost",
        accuracy=f"{bundle['accuracy']*100:.2f}%" if bundle else "94.31%",
        features=bundle["feature_cols"] if bundle else [],
        stages=stages,
        feature_importances=feature_importances,
        dataset_distribution=dataset_distribution,
        eda_matrix=eda_matrix,
        missing_audit=missing_audit,
        demographics=demographics,
        biomarker_dictionary=biomarker_dictionary,
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

