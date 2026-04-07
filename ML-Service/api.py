from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# SHAP — initialized once at startup
explainer = shap.TreeExplainer(model)

# Human-readable labels and thresholds for reason generation
FEATURE_THRESHOLDS = {
    "temperature": {"high": 320, "label_high": "High temperature", "label_low": "Low temperature", "unit": "K"},
    "vibration":   {"high": 1700, "label_high": "Excessive vibration", "label_low": "Low vibration", "unit": "rpm"},
    "pressure":    {"high": 45, "label_high": "High pressure", "label_low": "Low pressure", "unit": "Nm"},
    "runtime":     {"high": 1000, "label_high": "Extended tool wear", "label_low": "Low runtime", "unit": "min"},
}


def build_reason(feature, value, shap_val):
    """Convert a feature + SHAP value into a human-readable reason string."""
    meta = FEATURE_THRESHOLDS.get(feature, {})
    direction = "↑" if shap_val > 0 else "↓"
    impact = "increases" if shap_val > 0 else "decreases"

    if meta:
        label = meta["label_high"] if shap_val > 0 else meta["label_low"]
        unit = meta["unit"]
        return f"{direction} {label} ({value:.1f} {unit}) {impact} failure risk"
    return f"{direction} {feature} = {value:.2f} {impact} failure risk"


@app.route('/')
def home():
    return "🚀 Predictive Maintenance API Running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # ==============================
        # VALIDATE INPUT
        # ==============================
        required_fields = ["temperature", "vibration", "pressure", "runtime"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ==============================
        # PREPROCESS — must scale inputs (model was trained on scaled data)
        # ==============================
        sample = pd.DataFrame([{
            "temperature": float(data["temperature"]),
            "vibration":   float(data["vibration"]),
            "pressure":    float(data["pressure"]),
            "runtime":     float(data["runtime"]),
        }])

        sample_scaled = pd.DataFrame(
            scaler.transform(sample),
            columns=sample.columns
        )

        # ==============================
        # PREDICT
        # ==============================
        prob = float(model.predict_proba(sample_scaled)[0][1])

        # ==============================
        # STATUS THRESHOLDS
        # ==============================
        if prob > 0.65:
            status = "FAILURE"
        elif prob > 0.40:
            status = "WARNING"
        else:
            status = "SAFE"

        # ==============================
        # SHAP EXPLANATIONS
        # ==============================
        shap_values = explainer.shap_values(sample_scaled)

        # shap_values is list[2 arrays] for binary classification
        if isinstance(shap_values, list):
            vals = np.array(shap_values[1][0]).flatten()
        else:
            vals = np.array(shap_values[0]).flatten()

        features = list(sample.columns)
        raw_values = sample.iloc[0].to_dict()

        contributions = {features[i]: float(vals[i]) for i in range(len(features))}

        # ==============================
        # TOP REASONS — human-readable
        # ==============================
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        reasons = [
            build_reason(feat, raw_values[feat], shap_val)
            for feat, shap_val in sorted_contribs[:3]
        ]

        # Health score (0–100, inverse of failure probability)
        health_score = round((1 - prob) * 100, 1)

        return jsonify({
            "status": status,
            "probability": prob,
            "health_score": health_score,
            "contributions": contributions,
            "reasons": reasons,
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)