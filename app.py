import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Constants
g = 9.81
w = 1025
knots_to_ms = 0.514444

st.set_page_config(layout="wide")
st.title("Prediction of nCG and nBow for Planing Hulls (Physical Inputs)")

col1, col2 = st.columns(2)

with col1:
    st.header("Ship and Wave Inputs")
    L = st.number_input("Ship length, L [m]", value=24.384)
    beam = st.number_input("Chine beam, B [m]", value=7.3152)
    beta = st.number_input("Deadrise angle, β [deg]", min_value=5.0, max_value=45.0, value=15.0)
    disp = st.number_input("Displacement, Δ [kg]", value=84368.18)
    lcg_pct = st.number_input("Longitudinal center of gravity, LCG [%L]", min_value=0.0, max_value=100.0, value=42.0)
    h13 = st.number_input("Significant wave height, H1/3 [m]", min_value=0.0, value=1.40208)

    st.markdown("Enter a list of **speeds [knots]** (e.g. `24.5, 25.0, 26.2`):")
    speed_input = st.text_input("Speed list (V [knots])")

    st.markdown("Enter the corresponding **trim angles [deg]** (e.g. `3.0, 3.4, 3.6`):")
    trim_input = st.text_input("Trim list (τ [deg])")

    predict_button = st.button("Predict")

if predict_button:
    try:
        speeds = [float(x.strip()) for x in speed_input.split(",") if x.strip()]
        tau_list = [float(x.strip()) for x in trim_input.split(",") if x.strip()]

        if len(speeds) != len(tau_list):
            st.error(f"Please enter the same number of speeds and trim values. You entered {len(speeds)} speeds and {len(tau_list)} trims.")
        else:
            # Load models
            gpr_ncg_model = joblib.load('gpr_ncg_model.pkl')
            etr_nbow_model = joblib.load('etr_nbow_model.pkl')
            scaler_X1 = joblib.load('scaler_X1.pkl')
            scaler_X2 = joblib.load('scaler_X2.pkl')

            # Precomputed constants
            C_delta = disp / (w * beam ** 3)
            H13_B = h13 / beam

            results = []

            for v_knots, tau in zip(speeds, tau_list):
                v = v_knots * knots_to_ms
                Fn = v / np.sqrt(g * L)

                X1 = np.array([[beta, C_delta, lcg_pct, tau, Fn, H13_B]])
                X1_scaled = scaler_X1.transform(X1)
                pred_ncg = gpr_ncg_model.predict(X1_scaled)[0]

                X2 = np.array([[beta, C_delta, lcg_pct, tau, H13_B, pred_ncg]])
                X2_scaled = scaler_X2.transform(X2)
                pred_nbow = etr_nbow_model.predict(X2_scaled)[0]

                results.append({
                    "Speed [knots]": round(v_knots, 2),
                    "Trim [deg]": round(tau, 2),
                    "Froude Number (Fn)": round(Fn, 3),
                    "Predicted nCG [g]": round(pred_ncg, 3),
                    "Predicted nBow [g]": round(pred_nbow, 3)
                })

            df_results = pd.DataFrame(results)

            with col2:
                st.header("Prediction Results Table")
                st.success("Prediction completed.")
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

                df_plot = df_results.copy()
                df_plot["Speed [knots]"] = pd.to_numeric(df_plot["Speed [knots]"], errors="coerce")
                df_plot = df_plot.dropna(subset=["Speed [knots]"])
                df_plot = df_plot.sort_values("Speed [knots]")
                df_plot.set_index("Speed [knots]", inplace=True)

                st.line_chart(df_plot[["Predicted nCG [g]"]], use_container_width=True)
                st.line_chart(df_plot[["Predicted nBow [g]"]], use_container_width=True)

    except ValueError:
        st.error("Please enter only numeric values separated by commas in both fields.")
