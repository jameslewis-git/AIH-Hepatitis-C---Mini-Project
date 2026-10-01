"""
export_engine.py
Exports the trained HCV XGBoost model, imputer statistics, scaler parameters,
and clinical metadata into a compact, ultra-fast client-side JavaScript engine
(static/js/hcv_engine.js) for zero-latency, 100% offline Netlify deployment.
"""

import os
import json
import joblib

MODEL_PATH = "model/hcv_model.pkl"
OUTPUT_JS_PATH = "static/js/hcv_engine.js"


def export_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    class_labels = bundle["class_labels"]
    accuracy = round(float(bundle["accuracy"]) * 100, 1)
    model_name = bundle.get("model_name", "XGBoost")

    feat_to_idx = {name: i for i, name in enumerate(feature_cols)}
    booster = model.get_booster()
    dump = [json.loads(d) for d in booster.get_dump(dump_format="json")]

    def find_child(node, target_id):
        for c in node["children"]:
            if c["nodeid"] == target_id:
                return c
        raise ValueError(f"Child {target_id} not found in node {node.get('nodeid')}")

    def flatten_tree(node):
        nodes = []

        def traverse(n):
            idx = len(nodes)
            if "leaf" in n:
                nodes.append(round(n["leaf"], 8))
                return idx
            nodes.append(None)
            f_idx = feat_to_idx[n["split"]]
            thresh = round(n["split_condition"], 8)
            yes_node = find_child(n, n["yes"])
            no_node = find_child(n, n["no"])
            l_idx = traverse(yes_node)
            r_idx = traverse(no_node)
            nodes[idx] = [f_idx, thresh, l_idx, r_idx]
            return idx

        traverse(node)
        return nodes

    compact_trees = [flatten_tree(t) for t in dump]

    stage_info = {
        "Blood Donor": {
            "color": "#10b981",
            "rgb": "16,185,129",
            "icon": "✅",
            "desc": "No indication of liver disease. Lab values are within a healthy donor range."
        },
        "Suspect Blood Donor": {
            "color": "#f59e0b",
            "rgb": "245,158,11",
            "icon": "⚠️",
            "desc": "Borderline values. Not a confirmed donor profile: further clinical monitoring recommended."
        },
        "Hepatitis": {
            "color": "#f97316",
            "rgb": "249,115,22",
            "icon": "🔶",
            "desc": "Pattern consistent with active Hepatitis C infection. Clinical follow-up advised."
        },
        "Fibrosis": {
            "color": "#ef4444",
            "rgb": "239,68,68",
            "icon": "🔴",
            "desc": "Signs of liver fibrosis (scar tissue) consistent with progressive damage."
        },
        "Cirrhosis": {
            "color": "#9f1239",
            "rgb": "159,18,57",
            "icon": "🚨",
            "desc": "Advanced liver scarring. Urgent specialist hepatology consultation strongly recommended."
        }
    }

    reference_ranges = {
        "ALB": {"min": 35.0, "max": 50.0, "unit": "g/L"},
        "ALP": {"min": 40.0, "max": 130.0, "unit": "U/L"},
        "ALT": {"min": 7.0, "max": 56.0, "unit": "U/L"},
        "AST": {"min": 10.0, "max": 40.0, "unit": "U/L"},
        "BIL": {"min": 3.0, "max": 21.0, "unit": "µmol/L"},
        "CHE": {"min": 5.3, "max": 12.9, "unit": "kU/L"},
        "CHOL": {"min": 3.0, "max": 5.2, "unit": "mmol/L"},
        "CREA": {"min": 60.0, "max": 110.0, "unit": "µmol/L"},
        "GGT": {"min": 8.0, "max": 61.0, "unit": "U/L"},
        "PROT": {"min": 64.0, "max": 83.0, "unit": "g/L"},
    }

    engine_data = {
        "modelName": model_name,
        "accuracy": accuracy,
        "featureCols": feature_cols,
        "classLabels": class_labels,
        "imputerStats": [round(float(v), 4) for v in imputer.statistics_],
        "scalerMean": [round(float(v), 6) for v in scaler.mean_],
        "scalerScale": [round(float(v), 6) for v in scaler.scale_],
        "stageInfo": stage_info,
        "referenceRanges": reference_ranges,
        "trees": compact_trees
    }

    js_content = f"""/**
 * HCV Sentinel: Client-Side Machine Learning Diagnostic Engine
 * Project: Hepatitis C Detection and Staging using Machine Learning
 * Generated automatically by export_engine.py
 *
 * Implements 100% faithful client-side evaluation of trained XGBoost multi:softprob
 * ensemble trees, median imputation, and z-score standard scaling in <1ms.
 */

(function (root, factory) {{
  if (typeof define === 'function' && define.amd) {{
    define([], factory);
  }} else if (typeof module === 'object' && module.exports) {{
    module.exports = factory();
  }} else {{
    root.HCVEngine = factory();
  }}
}}(typeof self !== 'undefined' ? self : this, function () {{
  'use strict';

  var DATA = {json.dumps(engine_data, separators=(',', ':'))};

  function parseVal(v) {{
    if (v === null || v === undefined || v === '') return NaN;
    var n = parseFloat(v);
    return isNaN(n) ? NaN : n;
  }}

  function predict(inputDict) {{
    var featureCols = DATA.featureCols;
    var rawValues = [];
    var imputedMarkers = [];

    for (var i = 0; i < featureCols.length; i++) {{
      var col = featureCols[i];
      var raw = inputDict[col];
      var val = parseVal(raw);

      if (col === 'Sex') {{
        if (typeof raw === 'string') {{
          val = (raw.trim().toLowerCase() === 'female' || raw.trim() === '0') ? 0 : 1;
        }} else if (typeof raw === 'number') {{
          val = raw === 0 ? 0 : 1;
        }} else {{
          val = 1;
        }}
      }}

      if (isNaN(val)) {{
        imputedMarkers.push(col);
        val = DATA.imputerStats[i];
      }}

      rawValues.push(val);
    }}

    // Z-score standardization: (x - mean) / scale
    var scaled = new Float32Array(rawValues.length);
    for (var s = 0; s < rawValues.length; s++) {{
      scaled[s] = (rawValues[s] - DATA.scalerMean[s]) / DATA.scalerScale[s];
    }}

    // Evaluate 1500 compact trees
    // Multi:softprob base_score is 0.5 per class
    var numClasses = 5;
    var margins = new Float64Array([0.5, 0.5, 0.5, 0.5, 0.5]);
    var trees = DATA.trees;

    for (var t = 0; t < trees.length; t++) {{
      var tree = trees[t];
      var curr = 0;
      while (typeof tree[curr] !== 'number') {{
        var node = tree[curr];
        var featIdx = node[0];
        var thresh = node[1];
        var leftIdx = node[2];
        var rightIdx = node[3];
        curr = (scaled[featIdx] < thresh) ? leftIdx : rightIdx;
      }}
      margins[t % numClasses] += tree[curr];
    }}

    // Softmax
    var maxMargin = margins[0];
    for (var m = 1; m < numClasses; m++) {{
      if (margins[m] > maxMargin) maxMargin = margins[m];
    }}

    var sumExp = 0;
    var exps = new Float64Array(numClasses);
    for (var e = 0; e < numClasses; e++) {{
      exps[e] = Math.exp(margins[e] - maxMargin);
      sumExp += exps[e];
    }}

    var probs = [];
    var predIdx = 0;
    var maxP = 0;

    for (var p = 0; p < numClasses; p++) {{
      var probVal = exps[p] / sumExp;
      var stageName = DATA.classLabels[p] || DATA.classLabels[String(p)];
      probs.push({{
        stage: stageName,
        value: Math.round(probVal * 1000) / 10
      }});
      if (probVal > maxP) {{
        maxP = probVal;
        predIdx = p;
      }}
    }}

    // Sort descending by probability
    probs.sort(function (a, b) {{ return b.value - a.value; }});

    var winningLabel = DATA.classLabels[predIdx] || DATA.classLabels[String(predIdx)];
    var stageDetails = DATA.stageInfo[winningLabel] || {{
      color: '#dc2626',
      rgb: '220,38,38',
      icon: '📊',
      desc: 'Diagnosis pattern evaluated.'
    }};

    // Biomarker status flags
    var flags = [];
    var ranges = DATA.referenceRanges;
    for (var f = 0; f < featureCols.length; f++) {{
      var fName = featureCols[f];
      if (ranges[fName]) {{
        var userVal = rawValues[f];
        var r = ranges[fName];
        var status = 'normal';
        if (userVal > r.max) status = 'high';
        else if (userVal < r.min) status = 'low';

        flags.push({{
          marker: fName,
          value: userVal,
          status: status,
          normalMin: r.min,
          normalMax: r.max,
          unit: r.unit
        }});
      }}
    }}

    return {{
      label: winningLabel,
      confidence: Math.round(maxP * 1000) / 10,
      color: stageDetails.color,
      rgb: stageDetails.rgb,
      icon: stageDetails.icon,
      desc: stageDetails.desc,
      probs: probs,
      imputed_markers: imputedMarkers,
      imputedMarkers: imputedMarkers,
      flags: flags,
      modelName: DATA.modelName,
      accuracy: DATA.accuracy
    }};
  }}

  return {{
    predict: predict,
    getData: function () {{ return DATA; }}
  }};
}}));
"""

    os.makedirs(os.path.dirname(OUTPUT_JS_PATH), exist_ok=True)
    with open(OUTPUT_JS_PATH, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"Successfully generated client-side ML engine at {OUTPUT_JS_PATH}")
    print(f"File size: {round(os.path.getsize(OUTPUT_JS_PATH)/1024, 1)} KB")


if __name__ == "__main__":
    export_model()
