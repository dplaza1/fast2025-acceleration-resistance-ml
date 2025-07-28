import streamlit as st
import joblib
import numpy as np

# Load the trained models and scalers from disk
gpr_CG = joblib.load('gpr_ncg_model.pkl')
etr_Bow = joblib.load('etr_nbow_model.pkl')
scaler_X1 = joblib.load('scaler_X1.pkl')
scaler_X2 = joblib.load('scaler_X2.pkl')

# Set up the title and description for the Streamlit app
st.title("Prediction of nCG and nBow for Planing Hulls")

st.markdown(
    "Input the required parameters to predict **nCG** (vertical acceleration) and **nBow** (added resistance) "
    "for planing hulls using pre-trained machine learning models."
)

# Input fields for user to enter ship and wave parameters
beta = st.number_input("Beta [degrees]", value=0.0, help="Deadrise angle in the LCG (degrees)")
cv = st.number_input("Cv [-]", value=0.0, help=" Displacement Coefficient (dimensionless)")
lcg = st.number_input("LCG [%L]", value=0.0, help="Longitudinal center of gravity as (% of length)")
tao = st.number_input("Tao [degrees]", value=0.0, help="Trim angle (degrees)")
fn = st.number_input("Fn [-]", value=0.0, help="Froude number (dimensionless)")
h13b = st.number_input("H1/3 /b [-]", value=0.0, help="Significant wave height to beam ratio (dimensionless)")
ncg_input = st.number_input("ncg [g] for nBow prediction", value=0.0, help="Predicted or measured nCG value for nBow prediction")

if st.button("Predict"):
    # Prepare input data for nCG prediction and scale
    features_ncg = np.array([[beta, cv, lcg, tao, fn, h13b]])
    features_ncg_scaled = scaler_X1.transform(features_ncg)
    # Predict nCG using the Gaussian Process Regression model
    pred_ncg = gpr_CG.predict(features_ncg_scaled)[0]

    # Prepare input data for nBow prediction and scale
    features_nbow = np.array([[beta, cv, lcg, tao, h13b, ncg_input]])
    features_nbow_scaled = scaler_X2.transform(features_nbow)
    # Predict nBow using the Extra Trees Regressor model
    pred_nbow = etr_Bow.predict(features_nbow_scaled)[0]

    # Display the predictions to the user
    st.success(f"Predicted nCG: {pred_ncg:.4f}")
    st.success(f"Predicted nBow: {pred_nbow:.4f}")
