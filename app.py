import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravity (m/s^2)
w = 1025  # specific weight of water (kg/m^3)
knots_to_ms = 0.514444  # conversion factor knots to m/s

st.set_page_config(layout="wide")  # 💡 Para ancho completo

st.title("Prediction of nCG and nBow for Planing Hulls (Physical Inputs)")

# Divide la pantalla en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.header("Ship and Wave Inputs")
    L = st.number_input("Ship length, L [m]", value=24.384)
    beam = st.number_input("Chine beam, B [m]", value=7.3152)
    beta = st.number_input("Deadrise angle, β [deg]", min_value=5.0, max_value=45.0, value=15.0)
    disp = st.number_input("Displacement, Δ [kg]", value=84368.18)
    lcg_pct = st.number_input("Longitudinal center of gravity, LCG [%L]", min_value=0.0, max_value=100.0, value=42.0)
    tau = st.number_input("Trim angle, τ [deg]", min_value=0.0, value=3.6)
    h13 = st.number_input("Significant wave height, H1/3 [m]", min_value=0.0, value=1.40208)

    v_min, v_max = st.slider(
        "Select range of ship speeds (V [knots])",
        min_value=5.0, max_value=50.0, value=(25.0, 30.0), step=1.0
    )

    predict_button = st.button("Predict")

if predict_button:
    # Load models
    gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
    etr_nbow_model = joblib.load('etr_nbow_model.pkl')
    scaler_X1 = joblib.load('scaler_X1.pkl')
    scaler_X2 = joblib.load('scaler_X2.pkl')

    # Pre-calculated constants
    C_delta = disp / (w * beam ** 3)
    H13_B = h13 / beam

    results = []
    speeds = []
    fn_list = []
    ncg_list = []
    nbow_list = []

    for v_knots in np.arange(v_min, v_max + 0.01, 1.0):
        v = v_knots * knots_to_ms
        Fn = v / np.sqrt(g * L)

        X1 = np.array([[beta, C_delta, lcg_pct, tau, Fn, H13_B]])
        X1_scaled = scaler_X1.transform(X1)
        pred_ncg = gpr_ncg_model.predict(X1_scaled)[0]

        X2 = np.array([[beta, C_delta, lcg_pct, tau, H13_B, pred_ncg]])
        X2_scaled = scaler_X2.transform(X2)
        pred_nbow = etr_nbow_model.predict(X2_scaled)[0]

        speeds.append(v_knots)
        fn_list.append(Fn)
        ncg_list.append(pred_ncg)
        nbow_list.append(pred_nbow)

        results.append({
            "Speed [knots]": round(v_knots, 1),
            "Froude Number (Fn)": round(Fn, 3),
            "Predicted nCG [g]": round(pred_ncg, 3),
            "Predicted nBow [g]": round(pred_nbow, 3)
        })

    df_results = pd.DataFrame(results)

    with col2:
        st.header("Prediction Results Table")
        st.success("Prediction completed for selected speed range.")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

    with col1:
        st.header("Hydro Coefficients")
        st.write(f"**Displacement Coefficient (CΔ)**: {C_delta:.3f}")
        st.write(f"**LCG [%L]**: {lcg_pct:.2f}")
        st.write(f"**H1/3 to Beam Ratio (H1/3/B)**: {H13_B:.3f}")

    with col2:
        st.header("Graphs")

        fig1, ax1 = plt.subplots()
        ax1.plot(speeds, ncg_list, marker='o', label='nCG [g]')
        ax1.set_xlabel("Speed [knots]")
        ax1.set_ylabel("nCG [g]")
        ax1.set_title("nCG vs Speed")
        ax1.grid(True)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(speeds, nbow_list, marker='s', color='orange', label='nBow [g]')
        ax2.set_xlabel("Speed [knots]")
        ax2.set_ylabel("nBow [g]")
        ax2.set_title("nBow vs Speed")
        ax2.grid(True)
        st.pyplot(fig2)
