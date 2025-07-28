import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor
import joblib

# Load data
Data_ship1 = pd.read_csv('Acc_ncg.csv')
Data_ship2 = pd.read_csv('Acc_nBow.csv')

# Select features
Variables1 = ['Beta [degrees]', 'Cv [-]', 'LCG [%L]', 'Tao [degrees]', 'Fn [-]', 'H1/3 /b [-]']
X1 = Data_ship1[Variables1]

Variables2 = ['Beta [degrees]', 'Cv [-]', 'LCG [%L]', 'Tao [degrees]', 'H1/3 /b [-]', 'ncg [g]']
X2 = Data_ship2[Variables2]

# Set targets
Y = pd.DataFrame({
    'ncg [g]': Data_ship1['ncg [g]'],
    'nbow [g]': Data_ship2['nbow [g]']
})

# Scale features
scaler_X1 = MinMaxScaler()
normalized_X1 = scaler_X1.fit_transform(X1)

scaler_X2 = MinMaxScaler()
normalized_X2 = scaler_X2.fit_transform(X2)

# Split into train and test sets
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(
    normalized_X1, Y, test_size=0.2, random_state=42
)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
    normalized_X2, Y, test_size=0.2, random_state=42
)

Y_train1 = Y_train1.to_numpy()
Y_train2 = Y_train2.to_numpy()

# Define and train models
# Gaussian Process Regression for nCG
kernel = C(1.0, (1e-2, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 100)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1.0))
gpr_CG = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
gpr_CG.fit(X_train1, Y_train1[:, 0])

# Extra Trees Regressor for nBow
etr_Bow = ExtraTreesRegressor(n_estimators=50, random_state=42)
etr_Bow.fit(X_train2, Y_train2[:, 1])

# Save models and scalers
joblib.dump(gpr_CG, 'gpr_ncg_model.pkl')
joblib.dump(etr_Bow, 'etr_nbow_model.pkl')
joblib.dump(scaler_X1, 'scaler_X1.pkl')
joblib.dump(scaler_X2, 'scaler_X2.pkl')

print("Models and scalers have been saved successfully.")
