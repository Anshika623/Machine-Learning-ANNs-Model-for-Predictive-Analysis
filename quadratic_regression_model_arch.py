import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

#  Excel file
def historical_data(file_name, sheet_name='Sheet1'):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    return df

#  quadratic regression coefficients
def quadratic_coeffs(x, y):
    coeffs = np.polyfit(x, y, 2)  # Quadratic fit (degree 2)
    return coeffs
# R^2 and RMSE
def calculate_r2_rmse(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

#  combined quadratic regression MODEL
def combined_quadratic_coeffs(x1, x2, y):
    X = np.column_stack((x1, x2))
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler().fit(X_poly)
    X_poly_scaled = scaler.transform(X_poly)
    model = LinearRegression().fit(X_poly_scaled, y)
    return model, poly, scaler

# Excel file 
file_path = r'C:\Users\anshi\OneDrive\Desktop\scour_data.xlsx'
sheet_name = 'Sheet1'

# data
data = historical_data(file_path, sheet_name=sheet_name)

#  feature and target columns
f_cols = ['r/L (m)', 't/L', 'γM']
t_cols_limit = 'Limit Analysis (kN)'
t_cols_exp = 'Experimental (kN)'

# Extract data
x_l = data[f_cols[0]].values.tolist()  # r/L (m)
x_t = data[f_cols[1]].values.tolist()  # t/L
y_l = data[t_cols_limit].values.tolist()  # Limit Analysis (kN)
y_exp = data[t_cols_exp].values.tolist()  # Experimental (kN)

# INDIVIDUAL ANALYSIS
# quadratic coefficients
coeffs_r_la = quadratic_coeffs(x_l, y_l)
coeffs_r_exp = quadratic_coeffs(x_l, y_exp)
coeffs_t_la = quadratic_coeffs(x_t, y_l)
coeffs_t_exp = quadratic_coeffs(x_t, y_exp)

# New data
n_d = [0.25, 0.09, 22.7]
r_la_pred = np.polyval(coeffs_r_la, n_d[0])
r_exp_pred = np.polyval(coeffs_r_exp, n_d[0])
t_la_pred = np.polyval(coeffs_t_la, n_d[1])
t_exp_pred = np.polyval(coeffs_t_exp, n_d[1])

# Calculate R^2 and RMSE 
y_pred_r_la = np.polyval(coeffs_r_la, x_l)
r2_r_la, rmse_r_la = calculate_r2_rmse(y_l, y_pred_r_la)

y_pred_r_exp = np.polyval(coeffs_r_exp, x_l)
r2_r_exp, rmse_r_exp = calculate_r2_rmse(y_exp, y_pred_r_exp)

y_pred_t_la = np.polyval(coeffs_t_la, x_t)
r2_t_la, rmse_t_la = calculate_r2_rmse(y_l, y_pred_t_la)

y_pred_t_exp = np.polyval(coeffs_t_exp, x_t)
r2_t_exp, rmse_t_exp = calculate_r2_rmse(y_exp, y_pred_t_exp)

# Combined quadratic regression 
# Limit Analysis
model_la, poly_la, scaler_la = combined_quadratic_coeffs(x_l, x_t, y_l)
X_new_la = poly_la.transform([[n_d[0], n_d[1]]])
X_new_la_scaled = scaler_la.transform(X_new_la)
combined_la_pred = model_la.predict(X_new_la_scaled)[0]
X_poly_la = poly_la.transform(np.column_stack((x_l, x_t)))
X_poly_la_scaled = scaler_la.transform(X_poly_la)
y_pred_combined_la = model_la.predict(X_poly_la_scaled)
r2_combined_la, rmse_combined_la = calculate_r2_rmse(y_l, y_pred_combined_la)

# Combined quadratic regression 
# Experimental
model_exp, poly_exp, scaler_exp = combined_quadratic_coeffs(x_l, x_t, y_exp)
X_new_exp = poly_exp.transform([[n_d[0], n_d[1]]])
X_new_exp_scaled = scaler_exp.transform(X_new_exp)
combined_exp_pred = model_exp.predict(X_new_exp_scaled)[0]
X_poly_exp = poly_exp.transform(np.column_stack((x_l, x_t)))
X_poly_exp_scaled = scaler_exp.transform(X_poly_exp)
y_pred_combined_exp = model_exp.predict(X_poly_exp_scaled)
r2_combined_exp, rmse_combined_exp = calculate_r2_rmse(y_exp, y_pred_combined_exp)
 
# results

# individual analysis
print("Individual Analysis:")
print(f"r/L vs Limit Analysis: R^2 = {r2_r_la:.4f}, RMSE = {rmse_r_la:.4f}, Prediction = {r_la_pred:.2f}")
print(f"r/L vs Experimental: R^2 = {r2_r_exp:.4f}, RMSE = {rmse_r_exp:.4f}, Prediction = {r_exp_pred:.2f}")
print(f"t/L vs Limit Analysis: R^2 = {r2_t_la:.4f}, RMSE = {rmse_t_la:.4f}, Prediction = {t_la_pred:.2f}")
print(f"t/L vs Experimental: R^2 = {r2_t_exp:.4f}, RMSE = {rmse_t_exp:.4f}, Prediction = {t_exp_pred:.2f}")

# combined analysis
print("\nCombined Analysis:")
print(f"Combined Limit Analysis: R^2 = {r2_combined_la:.4f}, RMSE = {rmse_combined_la:.4f}, Prediction = {combined_la_pred:.2f}")
print(f"Combined Experimental: R^2 = {r2_combined_exp:.4f}, RMSE = {rmse_combined_exp:.4f}, Prediction = {combined_exp_pred:.2f}")

# quadratic regression equations
print("\nQuadratic Regression Equations:")
print(f"r/L vs Limit Analysis: y = {coeffs_r_la[0]:.4e}x^2 + {coeffs_r_la[1]:.4e}x + {coeffs_r_la[2]:.4e}")
print(f"r/L vs Experimental: y = {coeffs_r_exp[0]:.4e}x^2 + {coeffs_r_exp[1]:.4e}x + {coeffs_r_exp[2]:.4e}")
print(f"t/L vs Limit Analysis: y = {coeffs_t_la[0]:.4e}x^2 + {coeffs_t_la[1]:.4e}x + {coeffs_t_la[2]:.4e}")
print(f"t/L vs Experimental: y = {coeffs_t_exp[0]:.4e}x^2 + {coeffs_t_exp[1]:.4e}x + {coeffs_t_exp[2]:.4e}")

# Combined quadratic regression equation 
# Limit Analysis
coefs_combined_la = model_la.coef_
intercept_combined_la = model_la.intercept_
print("\nCombined Limit Analysis Equation:")
print(f"y = {intercept_combined_la:.4e} + {coefs_combined_la[1]:.4e}*x1 + {coefs_combined_la[2]:.4e}*x2 + {coefs_combined_la[3]:.4e}*x1^2 + {coefs_combined_la[4]:.4e}*x1*x2 + {coefs_combined_la[5]:.4e}*x2^2")

# Combined quadratic regression equation 
# Experimental
coefs_combined_exp = model_exp.coef_
intercept_combined_exp = model_exp.intercept_
print("\nCombined Experimental Equation:")
print(f"y = {intercept_combined_exp:.4e} + {coefs_combined_exp[1]:.4e}*x1 + {coefs_combined_exp[2]:.4e}*x2 + {coefs_combined_exp[3]:.4e}*x1^2 + {coefs_combined_exp[4]:.4e}*x1*x2 + {coefs_combined_exp[5]:.4e}*x2^2")

# Plot
plt.figure(figsize=(18, 12))

# Plot r/L vs Limit Analysis
plt.subplot(2, 2, 1)
plt.scatter(x_l, y_l, color='blue', label='Actual Limit Analysis Data')
x_vals_r_la = np.linspace(min(x_l), max(x_l), 100)
plt.plot(x_vals_r_la, np.polyval(coeffs_r_la, x_vals_r_la), color='red', label='Limit Analysis Quadratic ')
plt.xlabel('r/L (m)')
plt.ylabel('Limit Analysis (kN)')
plt.title('r/L vs Limit Analysis')
plt.legend()

# Plot r/L vs Experimental
plt.subplot(2, 2, 2)
plt.scatter(x_l, y_exp, color='green', label='Actual Experimental Data')
x_vals_r_exp = np.linspace(min(x_l), max(x_l), 100)
plt.plot(x_vals_r_exp, np.polyval(coeffs_r_exp, x_vals_r_exp), color='orange', label='Experimental Quadratic')
plt.xlabel('r/L (m)')
plt.ylabel('Experimental (kN)')
plt.title('r/L vs Experimental')
plt.legend()

# Plot t/L vs Limit Analysis
plt.subplot(2, 2, 3)
plt.scatter(x_t, y_l, color='purple', label='Actual Limit Analysis Data')
x_vals_t_la = np.linspace(min(x_t), max(x_t), 100)
plt.plot(x_vals_t_la, np.polyval(coeffs_t_la, x_vals_t_la), color='cyan', label='Limit Analysis Quadratic')
plt.xlabel('t/L (m)')
plt.ylabel('Limit Analysis (kN)')
plt.title('t/L vs Limit Analysis')
plt.legend()

# Plot t/L vs Experimental
plt.subplot(2, 2, 4)
plt.scatter(x_t, y_exp, color='brown', label='Actual Experimental Data')
x_vals_t_exp = np.linspace(min(x_t), max(x_t), 100)
plt.plot(x_vals_t_exp, np.polyval(coeffs_t_exp, x_vals_t_exp), color='yellow', label='Experimental Quadratic')
plt.xlabel('t/L (m)')
plt.ylabel('Experimental (kN)')
plt.title('t/L vs Experimental')
plt.legend()

plt.tight_layout()
plt.show()
