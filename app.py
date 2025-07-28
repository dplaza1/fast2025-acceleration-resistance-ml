import streamlit as st
import numpy as np
import joblib

# Constants
g = 9.81  # gravity (m/s^2)
w = 1025  # specific weight of water (kg/m^3)
knots_to_ms = 0.514444  # conversion factor knots to m/s

st.title("Prediction of nCG and nBow for Planing Hulls (Physical Inputs)")

st.markdown("""
Enter the physical parameters of the vessel and sea state. The app will calculate the required coefficients and predict nCG and nBow.
""")

# User physical inputs (in the requested order)
st.header("Input Ship and Wave Parameters")

L = st.number_input("Ship length, L [m]", min_value=5.0, value=30.0)
beam = st.number_input("Chine beam, B [m]", min_value=1.0, value=5.0)
beta = st.number_input("Deadrise angle, β [deg]", min_value=5.0, max_value=45.0, value=20.0)
disp = st.number_input("Displacement, Δ [kg]", min_value=1000.0, value=15000.0)
lcg_pct = st.number_input("Longitudinal center of gravity, LCG [%L]", min_value=0.0, max_value=100.0, value=38.0)
v_knots = st.number_input("Ship speed, V [knots]", min_value=1.0, value=20.0)
tau = st.number_input("Trim angle, τ [deg]", min_value=0.0, value=3.0)
h13 = st.number_input("Significant wave height, H1/3 [m]", min_value=0.1, value=1.0)

if st.button("Predict"):
    # Convert inputs
    v = v_knots * knots_to_ms  # Convert knots to m/s

    # Compute coefficients
    C_delta = disp / (w * beam ** 3)
    Fn = v / np.sqrt(g * L)
    H13_B = h13 / beam

    # Show calculated coefficients
    st.markdown("**Calculated Coefficients:**")
    st.write(f"Displacement Coefficient (CΔ): {C_delta:.3f}")
    st.write(f"LCG [%L]: {lcg_pct:.2f}")
    st.write(f"Froude Number (Fn): {Fn:.3f}")
    st.write(f"Significant Wave Height to Beam Ratio (H1/3/B): {H13_B:.3f}")

    # Load models and scalers
    gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
    etr_nbow_model = joblib.load('etr_nbow_model.pkl')
    scaler_X1 = joblib.load('scaler_X1.pkl')
    scaler_X2 = joblib.load('scaler_X2.pkl')

    # Prepare input arrays for ML models
    X1 = np.array([[beta, C_delta, lcg_pct, tau, Fn, H13_B]])
    X1_scaled = scaler_X1.transform(X1)
    pred_ncg = gpr_ncg_model.predict(X1_scaled)[0]

    # For nBow, match the training order: [beta, C_delta, lcg_pct, tau, H13_B, pred_ncg]
    X2 = np.array([[beta, C_delta, lcg_pct, tau, H13_B, pred_ncg]])
    X2_scaled = scaler_X2.transform(X2)
    pred_nbow = etr_nbow_model.predict(X2_scaled)[0]

    st.success(f"Predicted nCG: {pred_ncg:.3f} g")
    st.success(f"Predicted nBow: {pred_nbow:.3f} g")

    st.markdown("---")
    st.write("Inputs Used:")
    st.json({
        "L [m]": L,
        "B [m]": beam,
        "β [deg]": beta,
        "Δ [kg]": disp,
        "LCG [%L]": lcg_pct,
        "V [knots]": v_knots,
        "τ [deg]": tau,
        "H1/3 [m]": h13
    })
