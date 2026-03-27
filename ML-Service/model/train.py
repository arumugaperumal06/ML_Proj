import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
df = pd.read_csv('D:/ML/ML_Proj/data/ai4i2020.csv')

#renaming the columns

df = df.rename(columns={'Air temperature [K]': 'temperature',
    'Rotational speed [rpm]': 'vibration',
    'Torque [Nm]': 'pressure',
    'Tool wear [min]': 'runtime',
    'Machine failure': 'failure'
})

df = df[['temperature', 'vibration', 'pressure', 'runtime', 'failure']]

# print(df.isnull().sum())
df = df.dropna()
df = df.astype({
    'temperature': float,
    'vibration': float,
    'pressure': float,
    'runtime': float,
    'failure': int
})

print(df['failure'].value_counts())

X = df[['temperature', 'vibration', 'pressure', 'runtime']]
Y = df['failure']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model.fit(X_train, Y_train)

joblib.dump(model, 'model/model.pkl')

print("Model trained and saved as model.pkl")