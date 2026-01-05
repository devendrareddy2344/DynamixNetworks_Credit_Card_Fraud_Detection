import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Load model & scaler
# -------------------------------
@st.cache_resource
def load_pipeline():
    with open("credit_fraud_pipeline.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_pipeline()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is **Fraudulent or Normal** using an Extra Trees model.")

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("Enter Transaction Details")

time = st.number_input("Time", min_value=0.0, value=10000.0)
amount = st.number_input("Amount", min_value=0.0, value=100.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

# Arrange features in correct order
input_data = np.array([[
    time,
    *v_features,
    amount
]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"**Fraud Probability:** {probability:.4f}")
