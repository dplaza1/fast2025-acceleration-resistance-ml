import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
import joblib
import numpy as np

# Load data
Data_ship1 = pd.read_csv('Acc_ncg.csv')
Data_ship2 = pd.read_csv('Acc_nBow.csv')
Data_ship3 = pd.read_csv('AddRe.csv')

# Select features
Variables1 = ['Beta [degrees]', 'Cv [-]', 'LCG [%L]', 'Tao [degrees]', 'Fn [-]', 'H1/3 /b [-]']
X1 = Data_ship1[Variables1]
Y1 = Data_ship1['ncg [g]']

Variables2 = ['Beta [degrees]', 'Cv [-]', 'LCG [%L]', 'Tao [degrees]', 'H1/3 /b [-]', 'ncg [g]']
X2 = Data_ship2[Variables2]
Y2 = Data_ship2['nbow [g]']

Variables3 = ['Beta [degrees]', 'Cv [-]', 'Tao [degrees]', 'Fn [-]', 'H1/3 /b [-]']
X3 = Data_ship3[Variables3]
Y3 = Data_ship3['Ra/wb3 [-]']

# Scale features
scaler_X1 = MinMaxScaler()
X1_scaled = scaler_X1.fit_transform(X1)
scaler_X2 = MinMaxScaler()
X2_scaled = scaler_X2.fit_transform(X2)
scaler_X3 = MinMaxScaler()
X3_scaled = scaler_X3.fit_transform(X3)

# Split into train and test sets
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1_scaled, Y1, test_size=0.2, random_state=42)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2_scaled, Y2, test_size=0.2, random_state=42)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3_scaled, Y3, test_size=0.2, random_state=42)

# Define and train models
kernel = C(1.0, (1e-2, 1e4)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 100)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1.0))

# nCG
gpr_ncg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
gpr_ncg.fit(X_train1, Y_train1)
etr_ncg = ExtraTreesRegressor(n_estimators=50, random_state=42)
etr_ncg.fit(X_train1, Y_train1)

# nBow
gpr_nbow = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
gpr_nbow.fit(X_train2, Y_train2)
etr_nbow = ExtraTreesRegressor(n_estimators=50, random_state=42)
etr_nbow.fit(X_train2, Y_train2)

# Added Resistance
etr_addre = ExtraTreesRegressor(n_estimators=50, random_state=42)
etr_addre.fit(X_train3, Y_train3)
svr_addre = SVR(kernel='rbf', gamma='scale', C=10, epsilon=0.05)
svr_addre.fit(X_train3, Y_train3)

# Save models and scalers
joblib.dump(gpr_ncg, 'gpr_ncg_model.pkl')
joblib.dump(etr_ncg, 'etr_ncg_model.pkl')
joblib.dump(gpr_nbow, 'gpr_nbow_model.pkl')
joblib.dump(etr_nbow, 'etr_nbow_model.pkl')
joblib.dump(etr_addre, 'etr_addre_model.pkl')
joblib.dump(svr_addre, 'svr_addre_model.pkl')
joblib.dump(scaler_X1, 'scaler_X1.pkl')
joblib.dump(scaler_X2, 'scaler_X2.pkl')
joblib.dump(scaler_X3, 'scaler_X3.pkl')

print("Models and scalers have been saved successfully.")

