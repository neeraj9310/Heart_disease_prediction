import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("Logistic_heart_disease.pkl")
scaler = joblib.load("Scaler.pkl")
expected_col = joblib.load("columns.pkl")

st.title("Heart Disease Prediction Model")
st.markdown("Provide the following details")

# Inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80, 250, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    50, 700, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)
if st.button("Predict"):

    raw_input = {
        "Age": age,
        "Sex": 1 if sex == "M" else 0,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
    }

    # One-hot encoding
    raw_input[f"ChestPainType_{pain}"] = 1
    raw_input[f"RestingECG_{resting_ecg}"] = 1
    raw_input[f"ExerciseAngina_{exercise_angina}"] = 1
    raw_input[f"ST_Slope_{st_slope}"] = 1

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_col:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in training order
    input_df = input_df[expected_col]

    # Scale only the columns used during scaler.fit()
    scaler_cols = list(scaler.feature_names_in_)
    input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")