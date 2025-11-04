# main.py
# AI-Powered Crop Yield Prediction (SDG 2)
# Author: Amanuel Alemu Zewdu

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
# Make sure 'crop_yield_dataset.csv' is in the same folder as this script
data = pd.read_csv("crop_yield_dataset.csv")

# -----------------------------
# Step 2: Select Features & Target
# -----------------------------
# Features: rainfall, temperature, soil_quality, fertilizer_use
X = data[['rainfall', 'temperature', 'soil_quality', 'fertilizer_use']]
y = data['yield']

# -----------------------------
# Step 3: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 5: Train Random Forest Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
print("✅ Model Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# -----------------------------
# Step 7: Save Model
# -----------------------------
joblib.dump(model, "crop_yield_model.pkl")
print("✅ Model saved as 'crop_yield_model.pkl'")
