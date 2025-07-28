import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App title
st.title("Prediction of nCG and nBow for Planing Hulls")

st.markdown("""
This app predicts vertical acceleration (nCG) and added resistance (nBow) for planing hulls using machine learning models.
Simply enter the initial parameters. The app will first predict nCG and then automatically use that value to predict nBow.
""")

# Load models and scalers
gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
etr_nbow_model = joblib.load('etr_nbow_model.pkl')
scaler_X1 = joblib.load('scaler_X1.pkl')
scaler_X2 = joblib.load('scaler_X2.pkl')

# User input section
st.header("Enter ship and wave parameters")

beta = st.number_input("Beta [deg]", min_value=5.0, max_value=45.0, value=20.0)
cv = st.number_input("Cv [-]", min_value=0.8, value=1.3)
lcg = st.number_input("LCG [%L]", min_value=20.0, max_value=50.0, value=38.0)
tao = st.number_input("Tao [deg]", min_value=1.0, value=3.6)
fn = st.number_input("Froude Number [-]", min_value=0.4, value=1.0)
h13b = st.number_input("H1/3/b", min_value=0.10, value=0.43)

if st.button("Predict"):
    # Predict nCG
    X1 = np.array([[beta, cv, lcg, tao, fn, h13b]])
    X1_scaled = scaler_X1.transform(X1)
    pred_ncg = gpr_ncg_model.predict(X1_scaled)[0]

    # Predict nBow (using pred_ncg)
    X2 = np.array([[beta, cv, lcg, tao, fn, h13b, pred_ncg]])
    X2_scaled = scaler_X2.transform(X2)
    pred_nbow = etr_nbow_model.predict(X2_scaled)[0]

    st.success(f"Predicted nCG: {pred_ncg:.3f}")
    st.success(f"Predicted nBow: {pred_nbow:.3f}")

    st.markdown("---")
    st.write("Input parameters used:")
    st.json({
        "Beta": beta,
        "Cv": cv,
        "LCG": lcg,
        "Tao": tao,
        "Fn": fn,
        "H1/3/b": h13b
    })
