import joblib
import pandas as pd
import shap
import numpy as np

# -------- LOAD MODEL + SCALER --------
model   = joblib.load('model/model.pkl')
scaler  = joblib.load('model/scaler.pkl')   # ← FIX: was missing; predictions were on unscaled data

explainer = shap.TreeExplainer(model)


def explain_prediction(input_data):
    raw = pd.DataFrame([input_data])

    # ---- SCALE (FIX: must match training pipeline) ----
    sample = pd.DataFrame(
        scaler.transform(raw),
        columns=raw.columns
    )

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    print("\n🔍 Prediction:", pred)
    print("📊 Probability:", prob)

    # ---- SHAP ----
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]

    vals     = np.array(vals).flatten()
    features = list(sample.columns)

    contributions = {features[i]: float(vals[i]) for i in range(len(features))}

    sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    reasons = [k for k, v in sorted_features[:2]]

    print("\n📌 Contributions:")
    for k, v in contributions.items():
        print(f"  {k}: {v:+.4f}")

    print("\n🚨 Main Reasons:", reasons)

    return {
        "prediction":    int(pred),
        "probability":   float(prob),
        "contributions": contributions,
        "reasons":       reasons,
    }


# -------- TEST --------
if __name__ == "__main__":
    sample = {
        "temperature": 360,
        "vibration":   800,
        "pressure":    40,
        "runtime":     500,
    }
    explain_prediction(sample)