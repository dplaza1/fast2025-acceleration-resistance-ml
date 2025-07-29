import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

g = 9.81
w = 1025
knots_to_ms = 0.514444
m_to_ft = 3.28084

st.set_page_config(layout="wide")
st.title("Prediction of nCG and nBow for Planing Hulls ")

col1, col2 = st.columns(2)

with col1:
    L = st.number_input("Ship length, L [m]", value=1.79)
    beam = st.number_input("Chine beam, B [m]", value=0.382)
    beta = st.number_input("Deadrise angle, β [deg]", min_value=5.0, max_value=45.0, value=19.0)
    disp = st.number_input("Displacement, Δ [kg]", value=35.42)
    lcg_pct = st.number_input("Longitudinal center of gravity, LCG [%L]", min_value=0.0, max_value=100.0, value=38.0)
    h13 = st.number_input("Significant wave height, H1/3 [m]", min_value=0.0, value=0.102)

    speed_input = st.text_input("Speed list (knots)", value="4.806, 9.694, 14.504")
    trim_input = st.text_input("Trim list (deg)", value="3.9, 5.81, 4.38")
    predict_button = st.button("Predict")

    st.markdown("---")
    L_B = L / beam
    C_delta = disp / (w * beam ** 3)
    H13_B = h13 / beam

    coeff_data = {
        "Variable":      ["L/B [-]",    "β [°]",        "CΔ [-]",    "LCG [%L]",  "τ [°]",         "H1/3/B [-]"],
        "Value":         [f"{L_B:.3f}",  f"{beta:.1f}", f"{C_delta:.3f}", f"{lcg_pct:.1f}", "–", f"{H13_B:.3f}"],
        "Min–Max Range": ["4 – 9","10 – 30","0.384 – 1.20","28.6 – 45.7","2.0 – 9.2","0.215 – 0.750"],
    }
    df_coeff = pd.DataFrame(coeff_data)
    st.table(df_coeff)

if predict_button:
    try:
        speeds = [float(x.strip()) for x in speed_input.split(",") if x.strip()]
        tau_list = [float(x.strip()) for x in trim_input.split(",") if x.strip()]

        if len(speeds) != len(tau_list):
            st.error("Speeds and trims must have the same length.")
            st.stop()

        gpr_ncg = joblib.load('gpr_ncg_model.pkl')
        etr_nbow = joblib.load('etr_nbow_model.pkl')
        scaler1 = joblib.load('scaler_X1.pkl')
        scaler2 = joblib.load('scaler_X2.pkl')

        results = []
        L_ft = L * m_to_ft

        for v_knots, tau in zip(speeds, tau_list):
            v = v_knots * knots_to_ms
            Fn = v / np.sqrt(g * L)

            X1 = np.array([[beta, C_delta, lcg_pct, tau, Fn, H13_B]])
            y1_ml = gpr_ncg.predict(scaler1.transform(X1))[0]
            X2 = np.array([[beta, C_delta, lcg_pct, tau, H13_B, y1_ml]])
            y2_ml = etr_nbow.predict(scaler2.transform(X2))[0]

            Vk_sqrtL = v_knots / np.sqrt(L_ft)
            nCG_sav = (0.0104 *
                       (H13_B + 0.084) *
                       (tau / 4) *
                       (5/3 - beta/30) *
                       Vk_sqrtL**2 *
                       (L_B) /
                       C_delta)
            nBow_sav = nCG_sav * (1 + 3.8 * ((L_B - 2.25) / Vk_sqrtL))

            results.append({
                "Speed [knots]":      round(v_knots, 2),
                "Trim [deg]":         round(tau, 2),
                "Froude Number (Fn)": round(Fn, 3),
                "Pred nCG ML [g]":    round(y1_ml, 3),
                "Pred nBow ML [g]":   round(y2_ml, 3),
                "Pred nCG Sav [g]":   round(nCG_sav, 3),
                "Pred nBow Sav [g]":  round(nBow_sav, 3),
            })

        df_res = pd.DataFrame(results)

        with col2:
            st.header("Prediction Results Table")
            st.success("Prediction completed.")
            st.dataframe(df_res, use_container_width=True)

            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

            st.subheader("Graphs")
            speeds_np = df_res["Speed [knots]"].values
            ml_ncg = df_res["Pred nCG ML [g]"].values
            sav_ncg = df_res["Pred nCG Sav [g]"].values

            g1, g2 = st.columns(2)
            with g1:
                fig1, ax1 = plt.subplots(figsize=(3.5, 2.5), dpi=100)
                ax1.scatter(speeds_np, ml_ncg, color='blue', label='ML')
                ax1.scatter(speeds_np, sav_ncg, color='red', marker='x', label='Savitsky')
                ax1.set_xlabel("Speed [knots]", fontsize=8)
                ax1.set_ylabel("nCG [g]", fontsize=8)
                ax1.set_title("nCG vs Speed", fontsize=9)
                ax1.legend(fontsize=6)
                ax1.grid(True)
                fig1.tight_layout()
                st.pyplot(fig1)

            with g2:
                fig2, ax2 = plt.subplots(figsize=(3.5, 2.5), dpi=100)
                ml_nbow = df_res["Pred nBow ML [g]"].values
                sav_nbow = df_res["Pred nBow Sav [g]"].values
                ax2.scatter(speeds_np, ml_nbow, color='orange', label='ML')
                ax2.scatter(speeds_np, sav_nbow, color='green', marker='x', label='Savitsky')
                ax2.set_xlabel("Speed [knots]", fontsize=8)
                ax2.set_ylabel("nBow [g]", fontsize=8)
                ax2.set_title("nBow vs Speed", fontsize=9)
                ax2.legend(fontsize=6)
                ax2.grid(True)
                fig2.tight_layout()
                st.pyplot(fig2)

    except ValueError:
        st.error("Please enter only numeric values separated by commas in both fields.")

