import numpy as np
import pandas as pd
import joblib

model = joblib.load('model/model.pkl')
print("Model loaded successfully")

sample = pd.DataFrame([{
    'temperature': 340,   # extreme
    'vibration': 900,     # low
    'pressure': 20,       # low
    'runtime': 200        # low
}])
prediction = model.predict(sample)
prob = model.predict_proba(sample)

print(f"Prediction: {prediction[0]}, Probability: {prob[0][1]:.2f}")