# FAST2025 - Acceleration & Added Resistance Prediction for Planing Hulls


## Overview

This repository contains a machine learning-based app for predicting vertical accelerations (nCG, nBow) and added resistance for planing hulls. The models are trained on experimental datasets and allow users to input ship and wave characteristics for rapid predictions using modern ML regressors.

You can try the **interactive web app here**:  
👉 [https://fast2025-acceleration-resistance-ml-2slkqnmegvtdtvnqyf87qs.streamlit.app/]

---

## Features

- **Predict vertical acceleration at CG (nCG) and bow (nBow)**
- **Predict added resistance (Ra/wb³)**
- Compare ML predictions (Gaussian Process, Extra Trees, SVR) with classical Savitsky & Brown methods
- Interactive tables and download of results
- Visual graphs included for all outputs

---

## Getting Started (Python)

You can work with this repository and run the predictions locally in Python.

### **1. Requirements**

- Python 3.7+
- Packages listed in `requirements.txt` (install with `pip install -r requirements.txt`)

### **2. Files & Datasets**

The main dataset files are included:
- `Acc_nBow.csv`
- `Acc_ncg.csv`
- `AddRe.csv`
- --- Fast2025_PlazaEtal.ipynb --- Use this one to work in google colab.

All files must be in the same directory as `app.py` for correct operation.

---

## About This Study

This work is based on the research by the authors of the following paper:

**Plaza, D., Paredes, R., Begovic, E., Datla, R.**
"Accelerations and added-resistance predictions using machine learning for planing hulls"  

---

## Authors

- David Plaza  
- Ruben Paredes  
- Ermina Begovic  
- Raju Datla

---

## Contact

David Plaza - dplaza1@stevens.edu
