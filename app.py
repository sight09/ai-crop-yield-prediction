# app.py
# Streamlit App: AI-Powered Crop Yield Predictor
# Auto-trains model if not present
# Author: Amanuel Alemu Zewdu

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.title("🌾 AI-Powered Crop Yield Predictor")
st.write("Predict crop yield based on environmental conditions")

MODEL_FILE = "crop_yield_model.pkl"
DATA_FILE = "crop_yield_dataset.csv"

# -----------------------------
# Step 1: Check if model exists
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success("✅ Loaded existing model")
else:
    st.warning("⚠️ Model not found. Training model now...")

    # Load dataset
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found! Please make sure it exists.")
        st.stop()

    data = pd.read_csv(DATA_FILE)

    # Prepare features and target
    X = data[['rainfall', 'temperature', 'soil_quality', 'fertilizer_use']]
    y = data['yield']

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    st.success("✅ Model trained and saved successfully!")

# -----------------------------
# Step 2: User Input
# -----------------------------
rainfall = st.number_input("Rainfall (mm)", 0, 500, 200)
temperature = st.number_input("Temperature (°C)", 0, 50, 25)
soil_quality = st.slider("Soil Quality Index", 0.0, 1.0, 0.5)
fertilizer_use = st.number_input("Fertilizer (kg/ha)", 0, 300, 100)

# -----------------------------
# Step 3: Predict
# -----------------------------
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, soil_quality, fertilizer_use]])
    prediction = model.predict(input_data)
    st.success(f"🌱 Predicted Crop Yield: {prediction[0]:.2f} tons per hectare")
