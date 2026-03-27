import joblib
import pandas as pd
import shap
import numpy as np

# Load model
model = joblib.load('model/model.pkl')

# SHAP explainer
explainer = shap.TreeExplainer(model)


def explain_prediction(input_data):
    sample = pd.DataFrame([input_data])

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    print("\n🔍 Prediction:", pred)
    print("📊 Probability:", prob)

    # Get SHAP values
    shap_values = explainer.shap_values(sample)

    features = list(sample.columns)

    # -------- SAFE EXTRACTION --------
    try:
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values

        vals = np.array(vals)

        # Reduce dimensions safely
        while vals.ndim > 1:
            vals = vals[0]

        vals = vals.flatten()

    except Exception as e:
        print("⚠️ SHAP processing issue:", e)
        vals = np.zeros(len(features))

    # -------- MATCH FEATURE LENGTH SAFELY --------
    contributions = {}

    min_len = min(len(features), len(vals))

    for i in range(min_len):
        contributions[features[i]] = float(vals[i])

    # Fill missing features (if SHAP gave less values)
    if len(vals) < len(features):
        for i in range(len(vals), len(features)):
            contributions[features[i]] = 0.0

    # -------- PRINT OUTPUT --------
    print("\n📌 Feature Contributions:")
    for k, v in contributions.items():
        print(f"{k}: {v:.4f}")

    # -------- REASONS --------
    reasons = [k for k, v in contributions.items() if v > 0.1]

    print("\n🚨 Main Reasons:", reasons if reasons else "No major factors")

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "contributions": contributions,
        "reasons": reasons
    }


# TEST
if __name__ == "__main__":
    sample = {
        "temperature": 330,
        "vibration": 1800,
        "pressure": 50,
        "runtime": 1200
    }

    explain_prediction(sample)