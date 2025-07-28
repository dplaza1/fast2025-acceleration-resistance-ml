import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Título de la app
st.title("Predicción de nCG y nBow para cascos planeadores")

st.markdown("""
Esta app predice la aceleración vertical (nCG) y la resistencia adicional (nBow) para cascos planeadores usando modelos de machine learning.
Solo ingrese los parámetros iniciales. La app predecirá nCG y luego usará ese valor automáticamente para predecir nBow.
""")

# Cargar modelos y escaladores
gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
etr_nbow_model = joblib.load('etr_nbow_model.pkl')
scaler_X1 = joblib.load('scaler_X1.pkl')
scaler_X2 = joblib.load('scaler_X2.pkl')

# Inputs del usuario
st.header("Ingrese los datos del buque y la ola")

beta = st.number_input("Beta [deg]", min_value=5.0, max_value=45.0, value=20.0)
cv = st.number_input("Cv [-]", min_value=0.8, value=1.3)
lcg = st.number_input("LCG [%L]", min_value=20.0, max_value=50.0, value=38.0)
tao = st.number_input("Tao [deg]", min_value=1.0, value=3.6)
fn = st.number_input("Froude Number [--", min_value=0.4, value=1.0)
h13b = st.number_input("H1/3/b", min_value=0.10, value=0.43)

if st.button("Predecir"):
    # Predicción de nCG
    X1 = np.array([[beta, cv, lcg, tao, fn, h13b]])
    X1_scaled = scaler_X1.transform(X1)
    pred_ncg = gpr_ncg_model.predict(X1_scaled)[0]

    # Predicción de nBow (usa pred_ncg)
    X2 = np.array([[beta, cv, lcg, tao, fn, h13b, pred_ncg]])
    X2_scaled = scaler_X2.transform(X2)
    pred_nbow = etr_nbow_model.predict(X2_scaled)[0]

    st.success(f"Predicción de nCG: {pred_ncg:.3f}")
    st.success(f"Predicción de nBow: {pred_nbow:.3f}")

    st.markdown("---")
    st.write("Parámetros de entrada usados:")
    st.json({
        "Beta": beta,
        "Cv": cv,
        "LCG": lcg,
        "Tao": tao,
        "Fn": fn,
        "h13/b": h13b
    })
