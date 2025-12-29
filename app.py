import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load saved model artifacts
# -----------------------------
B = joblib.load("coefficients.pkl")
means = joblib.load("means.pkl")
stds = joblib.load("stds.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Heart Failure Predictor")

st.title("Heart Failure Risk Prediction")
st.write("Enter patient clinical data:")

# -----------------------------
# User input fields
# -----------------------------
inputs = []
for feature in feature_names:
    value = st.number_input(
        feature.replace("_", " ").title(),
        value=0.0
    )
    inputs.append(value)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    X_raw = np.array(inputs).reshape(1, -1)

    # Standardize using training statistics
    X_std = (X_raw - means) / stds

    # Add bias term
    X_matrix = np.hstack([np.ones((1, 1)), X_std])

    # Predict
    y_pred = X_matrix @ B
    score = float(y_pred[0][0])

    st.subheader("Prediction Result")
    st.write(f"Predicted Risk Score: **{score:.4f}**")

    if score >= 0.5:
        st.error("⚠ High Risk of Death Event")
    else:
        st.success("✅ Low Risk of Death Event")
