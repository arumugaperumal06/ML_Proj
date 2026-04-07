import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv('D:/ML/ML_Proj/data/ai4i2020.csv')

# ==============================
# RENAME
# ==============================
df = df.rename(columns={
    'Air temperature [K]': 'temperature',
    'Rotational speed [rpm]': 'vibration',
    'Torque [Nm]': 'pressure',
    'Tool wear [min]': 'runtime',
    'Machine failure': 'failure'
})

df = df[['temperature', 'vibration', 'pressure', 'runtime', 'failure']]

# ==============================
# CLEAN
# ==============================
df = df.dropna()

df = df.astype({
    'temperature': float,
    'vibration': float,
    'pressure': float,
    'runtime': float,
    'failure': int
})

print("\nOriginal Distribution:")
print(df['failure'].value_counts())

# ==============================
# BALANCE DATA
# ==============================
df_majority = df[df.failure == 0]
df_minority = df[df.failure == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

print("\nBalanced Distribution:")
print(df_balanced['failure'].value_counts())

# ==============================
# FEATURES
# ==============================
X = df_balanced[['temperature', 'vibration', 'pressure', 'runtime']]
Y = df_balanced['failure']

# ==============================
# SCALING (KEEP COLUMN NAMES)
# ==============================
scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# ==============================
# SPLIT
# ==============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# ==============================
# MODEL
# ==============================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, Y_train)

# ==============================
# SAVE
# ==============================
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n✅ Model & scaler saved!")

# ==============================
# FEATURE IMPORTANCE
# ==============================
print("\nFeature Importance:")
for name, val in zip(X.columns, model.feature_importances_):
    print(f"{name}: {val:.4f}")