# 🌾 AI-Powered Crop Yield Prediction for Zero Hunger (SDG 2)

### 🚀 Overview
This project uses Artificial Intelligence and Software Engineering principles to help farmers predict crop yields and improve agricultural planning.  
By leveraging real-time environmental data, it supports **UN SDG 2: Zero Hunger** — promoting sustainable agriculture and food security.

---

## 🎯 SDG Focus

- **Goal:** SDG 2 – Zero Hunger  
- **Problem:** Smallholder farmers often struggle to predict crop yield due to unpredictable weather and soil conditions.  
- **Solution:** Train an AI regression model to forecast crop yield using rainfall, temperature, soil, and fertilizer data.

---

## 🤖 AI & Software Engineering Approach

| Concept | Application |
|----------|--------------|
| **Automation** | Collect data and preprocess automatically using Python scripts |
| **Testing** | Use unit tests to ensure prediction validity |
| **Scalability** | Modular design with Streamlit for easy deployment |
| **Version Control** | Git + GitHub for collaboration |
| **CI/CD** | GitHub Actions to automate testing and updates |

---

## 🧰 Tools & Frameworks
- **Languages:** Python  
- **AI/ML:** Scikit-learn, Pandas, NumPy  
- **Deployment:** Streamlit  
- **Visualization:** Matplotlib, Seaborn  
- **Version Control:** Git, GitHub  
- **Testing:** Unittest / Pytest  
- **Dataset:** `crop_yield_dataset.csv` (synthetic or Kaggle dataset)

---

## 💻 Streamlit App

Create a file named `app.py`. The app:

- Loads the trained model `crop_yield_model.pkl`.  
- **Automatically trains the model** if the file is missing using `crop_yield_dataset.csv`.  
- Predicts crop yield based on user inputs.  

```bash
streamlit run app.py

User Inputs:

    Rainfall (mm)

    Temperature (°C)

    Soil Quality Index (0.0–1.0)

    Fertilizer Use (kg/ha)

🧪 Unit Test Example

Create test_model.py:

import joblib
import numpy as np

model = joblib.load("crop_yield_model.pkl")

def test_prediction_is_positive():
    pred = model.predict([[200, 25, 0.7, 120]])[0]
    assert pred > 0, "Prediction should be positive"

Run test:

pytest test_model.py

🔍 Ethical & Sustainability Checks
Concern	Action
Bias	Use diverse global data to avoid overfitting to one region.
Energy Efficiency	Lightweight Random Forest model, not deep neural networks.
Accessibility	Open-source Streamlit web app for public use.
📈 Results Example
Metric	Value
R² Score	0.89
MAE	0.23
Inference Speed	~0.01s per sample
🌍 Impact Summary

✅ Supports SDG 2: Zero Hunger
✅ Enables data-driven agriculture
✅ Reduces crop loss through predictive insights
✅ Encourages ethical, sustainable AI development
🧠 Reflection

How does it align with SDG 2?
Promotes sustainable farming through predictive analytics.

What ethical issues exist?
Bias and limited access; mitigated via open datasets and transparency.

How do software engineering practices help?
Testing, CI/CD, and modular design make it scalable and reliable.
```

## 📊 Pitch Deck

You can view the project pitch slides here:  
[AI-Powered Crop Yield Prediction — Pitch Deck](https://docs.google.com/presentation/d/1YJSQDroHef7e56Zs5LG8_ffxywVoK79seUZ3VG3DtDc/edit?slide=id.g3a033ef89da_0_517#slide=id.g3a033ef89da_0_517)  

> Note: The slides are set to **view-only** to prevent editing.


```

👨‍💻 Contributors

    Amanuel Alemu Zewdu

📜 License

MIT License © 2025


---

### ✅ Folder Structure for GitHub

📦 ai-crop-yield-prediction
┣ 📜 README.md
┣ 📜 app.py ← Streamlit app with auto-training
┣ 📜 test_model.py ← Unit tests
┣ 📜 crop_yield_dataset.csv
┗ 📜 crop_yield_model.pkl ← auto-created after first app run


---
