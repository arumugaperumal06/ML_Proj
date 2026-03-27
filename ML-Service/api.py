from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

# SHAP explainer
explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return "Predictive Maintenance API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input
    sample = pd.DataFrame([data])

    # Prediction
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    # SHAP explanation
    shap_values = explainer.shap_values(sample)

    features = sample.columns
    contributions = {}

    for i, val in enumerate(shap_values[1][0]):
        contributions[features[i]] = float(val)

    # Extract reasons
    reasons = [f for f, v in contributions.items() if v > 0.1]

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob),
        "contributions": contributions,
        "reasons": reasons
    })

if __name__ == '__main__':
    app.run(debug=True)