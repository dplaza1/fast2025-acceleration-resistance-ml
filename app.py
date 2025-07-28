import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Constants
g = 9.81
w = 1025
knots_to_ms = 0.514444

st.set_page_config(layout="wide")
st.title("Prediction of nCG and nBow for Planing Hulls (Physical Inputs)")

col1, col2 = st.columns(2)

with col1:
    st.header("Parametric Coefficients")
    # Precompute the three coefficients that depend only on inputs
    L = st.number_input("Ship length, L [m]", value=24.384)
    beam = st.number_input("Chine beam, B [m]", value=7.3152)
    beta = st.number_input("Deadrise angle, β [deg]", min_value=5.0, max_value=45.0, value=15.0)
    disp = st.number_input("Displacement, Δ [kg]", value=84368.18)
    lcg_pct = st.number_input("Longitudinal center of gravity, LCG [%L]", min_value=0.0, max_value=100.0, value=42.0)
    h13 = st.number_input("Significant wave height, H1/3 [m]", min_value=0.0, value=1.40208)

    C_delta = disp / (w * beam ** 3)
    H13_B = h13 / beam
    L_B = L / beam

    coeff_data = {
        "Variable":      ["L/B [-]",    "β [°]",          "CΔ [-]",    "LCG [%L]",  "τ [°]",           "H1/3 / B [-]"],
        "Value":         [f"{L_B:.3f}",  f"{beta:.1f}",   f"{C_delta:.3f}", f"{lcg_pct:.1f}", "–",          f"{H13_B:.3f}"],
        "Min–Max Range": ["4 – 9","10 – 30","0.384 – 1.200","28.6 – 45.7","2.0 – 9.2","0.215 – 0.750"]
    }
    df_coeff = pd.DataFrame(coeff_data)
    st.table(df_coeff)

    st.markdown("---")
    st.markdown("### Enter a list of **speeds [knots]**:")
    speed_input = st.text_input("Speed list (V [knots])", value="25.4, 38.1, 50.8")

    st.markdown("### Enter the corresponding **trim angles [deg]**:")
    trim_input = st.text_input("Trim list (τ [deg])", value="3.6, 3.5, 5.7")

    predict_button = st.button("Predict")

if predict_button:
    try:
        # parse
        speeds   = [float(x.strip()) for x in speed_input.split(",") if x.strip()]
        tau_list = [float(x.strip()) for x in trim_input.split(",") if x.strip()]

        if len(speeds) != len(tau_list):
            st.error("Speeds and trims must have the same length.")
            st.stop()

        # load models
        gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
        etr_nbow_model = joblib.load('etr_nbow_model.pkl')
        scaler_X1 = joblib.load('scaler_X1.pkl')
        scaler_X2 = joblib.load('scaler_X2.pkl')

        # compute results
        results = []
        for v_knots, tau in zip(speeds, tau_list):
            v = v_knots * knots_to_ms
            Fn = v / np.sqrt(g * L)

            X1 = np.array([[beta, C_delta, lcg_pct, tau, Fn, H13_B]])
            pred_ncg  = gpr_ncg_model.predict(scaler_X1.transform(X1))[0]

            X2 = np.array([[beta, C_delta, lcg_pct, tau, H13_B, pred_ncg]])
            pred_nbow = etr_nbow_model.predict(scaler_X2.transform(X2))[0]

            results.append({
                "Speed [knots]":        round(v_knots, 2),
                "Trim [deg]":           round(tau, 2),
                "Froude Number (Fn)":   round(Fn, 3),
                "Predicted nCG [g]":    round(pred_ncg, 3),
                "Predicted nBow [g]":   round(pred_nbow, 3)
            })

        df_results = pd.DataFrame(results)

        with col2:
            st.header("Prediction Results Table")
            st.success("Prediction completed.")
            st.dataframe(df_results, use_container_width=True)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

            st.subheader("Graphs")
            speeds_np = np.array(df_results["Speed [knots]"])
            ncg_np    = np.array(df_results["Predicted nCG [g]"])
            nbow_np   = np.array(df_results["Predicted nBow [g]"])

            g1, g2 = st.columns(2)

            with g1:
                fig1, ax1 = plt.subplots(figsize=(3.5, 2.5), dpi=100)
                ax1.scatter(speeds_np, ncg_np, color='blue')
                ax1.set_xlabel("Speed [knots]", fontsize=8)
                ax1.set_ylabel("nCG [g]", fontsize=8)
                ax1.set_title("nCG vs Speed", fontsize=9)
                ax1.tick_params(labelsize=7)
                ax1.grid(True)
                fig1.tight_layout()
                st.pyplot(fig1)

            with g2:
                fig2, ax2 = plt.subplots(figsize=(3.5, 2.5), dpi=100)
                ax2.scatter(speeds_np, nbow_np, color='orange')
                ax2.set_xlabel("Speed [knots]", fontsize=8)
                ax2.set_ylabel("nBow [g]", fontsize=8)
                ax2.set_title("nBow vs Speed", fontsize=9)
                ax2.tick_params(labelsize=7)
                ax2.grid(True)
                fig2.tight_layout()
                st.pyplot(fig2)

    except ValueError:
        st.error("Please enter only numeric values separated by commas in both fields.")

        st.error("Please enter only numeric values separated by commas in both fields.")
