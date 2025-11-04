import joblib
import numpy as np

model = joblib.load("crop_yield_model.pkl")

def test_prediction_is_positive():
    pred = model.predict([[200, 25, 0.7, 120]])[0]
    assert pred > 0, "Prediction should be positive"
