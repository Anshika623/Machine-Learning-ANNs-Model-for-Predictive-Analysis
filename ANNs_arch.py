import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Excel
file_path = r'C:\Users\anshi\OneDrive\Desktop\scour_data.xlsx'  # replace with your file path
df = pd.read_excel(file_path)

# Print column names to verify
print("Column names in the DataFrame:", df.columns)

# Drop the γI column
if 'γI' in df.columns:
    df = df.drop(columns=['γI'])

# Prepare features (X) and targets (y)
X = df[['r/L (m)', 't/L', 'γM']].values  # Ensure these column names match exactly
y_limit_analysis = df['Limit Analysis (kN)'].values
y_experimental = df['Experimental (kN)'].values

# Split the data into training and test sets
X_train, X_test, y_train_limit, y_test_limit = train_test_split(X, y_limit_analysis, test_size=0.2, random_state=42)
X_train, X_test, y_train_exp, y_test_exp = train_test_split(X, y_experimental, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train, val in kfold.split(X_train, y_train_limit):
    model_limit = build_model()
    model_limit.fit(X_train[train], y_train_limit[train], epochs=100, batch_size=10, validation_data=(X_train[val], y_train_limit[val]))

# Evaluate the model for Limit Analysis
y_pred_limit = model_limit.predict(X_test)
loss, mae = model_limit.evaluate(X_test, y_test_limit)
r2_limit = r2_score(y_test_limit, y_pred_limit)
rmse_limit = np.sqrt(mean_squared_error(y_test_limit, y_pred_limit))
print(f'Limit Analysis Model - Loss: {loss}, MAE: {mae}, R²: {r2_limit}, RMSE: {rmse_limit}')

# Cross-validation for Experimental
for train, val in kfold.split(X_train, y_train_exp):
    model_exp = build_model()
    model_exp.fit(X_train[train], y_train_exp[train], epochs=100, batch_size=10, validation_data=(X_train[val], y_train_exp[val]))

# Evaluate the model for Experimental
y_pred_exp = model_exp.predict(X_test)
loss, mae = model_exp.evaluate(X_test, y_test_exp)
r2_exp = r2_score(y_test_exp, y_pred_exp)
rmse_exp = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
print(f'Experimental Model - Loss: {loss}, MAE: {mae}, R²: {r2_exp}, RMSE: {rmse_exp}')

# Predict with new data
new_data = np.array([[0.3, 0.08, 22.0]])  # Example new data
new_data_scaled = scaler.transform(new_data)
prediction_limit = model_limit.predict(new_data_scaled)
prediction_exp = model_exp.predict(new_data_scaled)

# Clip predictions to ensure positivity and reasonable scale
prediction_limit_clipped = np.clip(prediction_limit, 0, None)
prediction_exp_clipped = np.clip(prediction_exp, 0, None)

print(f'Prediction for Limit Analysis: {prediction_limit_clipped}')
print(f'Prediction for Experimental: {prediction_exp_clipped}')

# Plotting the graphs
plt.figure(figsize=(12, 10))

# Plot r/L vs Limit Analysis and Experimental
plt.subplot(2, 2, 1)
plt.scatter(X_test[:, 0], y_test_limit, color='blue', label='Actual Limit Analysis Data')
plt.scatter(X_test[:, 0], y_pred_limit.flatten(), color='red', label='Predicted Limit Analysis Data')
plt.xlabel('r/L (m)')
plt.ylabel('Limit Analysis (kN)')
plt.title('r/L vs Limit Analysis')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(X_test[:, 0], y_test_exp, color='green', label='Actual Experimental Data')
plt.scatter(X_test[:, 0], y_pred_exp.flatten(), color='orange', label='Predicted Experimental Data')
plt.xlabel('r/L (m)')
plt.ylabel('Experimental (kN)')
plt.title('r/L vs Experimental')
plt.legend()

# Plot t/L vs Limit Analysis and Experimental
plt.subplot(2, 2, 3)
plt.scatter(X_test[:, 1], y_test_limit, color='purple', label='Actual Limit Analysis Data')
plt.scatter(X_test[:, 1], y_pred_limit.flatten(), color='cyan', label='Predicted Limit Analysis Data')
plt.xlabel('t/L (m)')
plt.ylabel('Limit Analysis (kN)')
plt.title('t/L vs Limit Analysis')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(X_test[:, 1], y_test_exp, color='brown', label='Actual Experimental Data')
plt.scatter(X_test[:, 1], y_pred_exp.flatten(), color='yellow', label='Predicted Experimental Data')
plt.xlabel('t/L (m)')
plt.ylabel('Experimental (kN)')
plt.title('t/L vs Experimental')
plt.legend()

plt.tight_layout()
plt.show()
