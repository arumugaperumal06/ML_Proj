import time
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model
model = joblib.load('model/model.pkl')

# CSV file path
file_path = '../data/live_data.csv'

# Create CSV if not exists
if not os.path.exists(file_path):
    df_init = pd.DataFrame(columns=[
        'temperature', 'vibration', 'pressure', 'runtime',
        'prediction', 'probability'
    ])
    df_init.to_csv(file_path, index=False)

# Initial values
temperature = 300
vibration = 1200
pressure = 30
runtime = 100

# Graph lists
time_step = []
runtime_list = []
prob_list = []

# Previous probability (for trend)
prev_prob = 0

plt.ion()

while True:
    # --- SIMULATION LOGIC ---
    temperature += random.uniform(-1, 2)
    vibration += random.uniform(-50, 80)
    pressure += random.uniform(-2, 3)
    runtime += random.uniform(5, 15)

    # Clamp values
    temperature = max(280, min(350, temperature))
    vibration = max(800, min(2200, vibration))
    pressure = max(10, min(70, pressure))

    # Random spike
    if random.random() < 0.1:
        vibration += random.uniform(200, 400)
        print("⚡ Sudden vibration spike!")

    # Create input
    sample = pd.DataFrame([{
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'runtime': runtime
    }])

    # Prediction
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    # Status levels
    if prob < 0.3:
        status = "SAFE"
    elif prob < 0.7:
        status = "⚠️ WARNING"
    else:
        status = "🚨 FAILURE RISK"

    # Reasoning (simple XAI)
    reasons = []
    if temperature > 320:
        reasons.append("High Temp")
    if vibration > 1700:
        reasons.append("High Vibration")
    if pressure > 45:
        reasons.append("High Pressure")
    if runtime > 1000:
        reasons.append("Long Runtime")

    # Trend detection
    trend = "📈 Increasing Risk" if prob > prev_prob else "📉 Stable"
    prev_prob = prob

    # Health score
    health = 100 - (prob * 100)

    # Print output
    print(f"""
Temp: {temperature:.2f} | Vib: {vibration:.2f} | Pressure: {pressure:.2f} | Runtime: {runtime:.2f}
Status: {status} | Risk: {prob:.2f} | Health: {health:.2f}%
Reason: {", ".join(reasons) if reasons else "Normal"}
{trend}
""")

    # Save to CSV
    new_row = pd.DataFrame([{
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'runtime': runtime,
        'prediction': pred,
        'probability': prob
    }])
    new_row.to_csv(file_path, mode='a', header=False, index=False)

    # --- GRAPH UPDATE ---
    time_step.append(len(time_step))
    runtime_list.append(runtime)
    prob_list.append(prob)

    plt.clf()

    # Runtime graph
    plt.subplot(2, 1, 1)
    plt.plot(time_step, runtime_list)
    plt.title("Runtime Over Time")

    # Probability graph
    plt.subplot(2, 1, 2)
    plt.plot(time_step, prob_list)
    plt.title("Failure Probability Over Time")

    plt.pause(0.1)

    # Maintenance reset
    if runtime > 1200:
        print("🔧 Maintenance performed → resetting system\n")
        temperature = 300
        vibration = 1200
        pressure = 30
        runtime = 100

    # Delay
    time.sleep(2)