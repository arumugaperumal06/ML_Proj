import time
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')   # ← FIX: was missing; caused stuck/wrong predictions

# CSV file path
file_path = '../data/live_data.csv'

# Create CSV if not exists
if not os.path.exists(file_path):
    df_init = pd.DataFrame(columns=[
        'temperature', 'vibration', 'pressure', 'runtime',
        'prediction', 'probability'
    ])
    df_init.to_csv(file_path, index=False)

# Initial sensor values
temperature = 300
vibration = 1200
pressure = 30
runtime = 100

# Graph history
time_step = []
runtime_list = []
prob_list = []

prev_prob = 0

plt.ion()

while True:
    # --- SIMULATE SENSOR DRIFT ---
    temperature += random.uniform(-1, 2)
    vibration   += random.uniform(-50, 80)
    pressure    += random.uniform(-2, 3)
    runtime     += random.uniform(5, 15)

    # Clamp to realistic ranges
    temperature = max(280, min(350, temperature))
    vibration   = max(800, min(2200, vibration))
    pressure    = max(10,  min(70, pressure))

    # Random spike
    if random.random() < 0.1:
        vibration += random.uniform(200, 400)
        print("⚡ Sudden vibration spike!")

    # --- BUILD SAMPLE ---
    raw_sample = pd.DataFrame([{
        'temperature': temperature,
        'vibration':   vibration,
        'pressure':    pressure,
        'runtime':     runtime,
    }])

    # --- SCALE INPUT (FIX: apply same scaler used during training) ---
    scaled_sample = pd.DataFrame(
        scaler.transform(raw_sample),
        columns=raw_sample.columns
    )

    # --- PREDICT ---
    pred = model.predict(scaled_sample)[0]
    prob = model.predict_proba(scaled_sample)[0][1]

    # --- STATUS ---
    if prob < 0.3:
        status = "SAFE"
    elif prob < 0.7:
        status = "⚠️ WARNING"
    else:
        status = "🚨 FAILURE RISK"

    # --- SIMPLE RULE-BASED REASONS ---
    reasons = []
    if temperature > 320:
        reasons.append("High Temp")
    if vibration > 1700:
        reasons.append("High Vibration")
    if pressure > 45:
        reasons.append("High Pressure")
    if runtime > 1000:
        reasons.append("Long Runtime")

    trend   = "📈 Increasing Risk" if prob > prev_prob else "📉 Stable"
    health  = 100 - (prob * 100)
    prev_prob = prob

    print(f"""
Temp: {temperature:.2f} | Vib: {vibration:.2f} | Pressure: {pressure:.2f} | Runtime: {runtime:.2f}
Status: {status} | Risk: {prob:.2f} | Health: {health:.2f}%
Reason: {", ".join(reasons) if reasons else "Normal"}
{trend}
""")

    # --- APPEND TO CSV ---
    new_row = pd.DataFrame([{
        'temperature': temperature,
        'vibration':   vibration,
        'pressure':    pressure,
        'runtime':     runtime,
        'prediction':  pred,
        'probability': prob,
    }])
    new_row.to_csv(file_path, mode='a', header=False, index=False)

    # --- GRAPH UPDATE ---
    time_step.append(len(time_step))
    runtime_list.append(runtime)
    prob_list.append(prob)

    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(time_step, runtime_list, color='steelblue')
    plt.title("Runtime Over Time")
    plt.ylabel("Runtime (min)")

    plt.subplot(2, 1, 2)
    plt.plot(time_step, prob_list, color='crimson')
    plt.axhline(0.65, color='red',    linestyle='--', linewidth=0.8, label='Failure threshold')
    plt.axhline(0.40, color='orange', linestyle='--', linewidth=0.8, label='Warning threshold')
    plt.title("Failure Probability Over Time")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()
    plt.pause(0.1)

    # --- MAINTENANCE RESET ---
    if runtime > 1200:
        print("🔧 Maintenance performed → resetting system\n")
        temperature = 300
        vibration   = 1200
        pressure    = 30
        runtime     = 100

    time.sleep(2)