import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  mean
def mean(values):
    return sum(values) / len(values)

# variance
def variance(values, mean_value):
    return sum((x - mean_value) ** 2 for x in values)

# co-variance
def co_var(x, mean_x, y, mean_y):
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))

# coefficients (b0, b1)
def coeffs(x, y):
    mean_x, mean_y = mean(x), mean(y)
    b1 = co_var(x, mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    return b0, b1

# plot regression
def plot_regression(x, y, b0, b1, new_data, xlabel, ylabel, title):
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot([min(x), max(x)], [b0 + b1 * min(x), b0 + b1 * max(x)], color='red', label='Regression Line')
    plt.scatter(new_data, b0 + b1 * new_data, color='green', s=100, label='Predicted Point', edgecolors='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

# linear regression
def linear_regression(x, y, b0, b1, new_data, xlabel, ylabel, title):
    y_pred = [b0 + b1 * xi for xi in x]
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    equation = f'y = {b0:.2f} + {b1:.2f}x'
    prediction = b0 + b1 * new_data
    print(f"{title}:")
    print(f"  R^2: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Linear Regression Equation: {equation}")
    print(f"  Prediction for new data ({xlabel}={new_data}): {prediction:.2f}")
    return r2, rmse, prediction, y_pred

#  multiple linear regression
def combined_reg(X, y, n_d, title):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    equation = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f} * r/L + {model.coef_[1]:.2f} * t/L'
    pred = model.predict([n_d])
    print(f"{title}:")
    print(f"  R^2: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Linear Regression Equation: {equation}")
    print(f"  Prediction for new data (r/L={n_d[0]}, t/L={n_d[1]}): {pred[0]:.2f}")
    return r2, rmse, pred[0], y_pred

# Excel
file_path = r'C:\Users\anshi\OneDrive\Desktop\scour_data.xlsx'
sheet_name = 'Sheet1'

# data
data = pd.read_excel(file_path, sheet_name=sheet_name)

#  feature and target
f_cols = ['r/L (m)', 't/L', 'γM']
t_cols_limit = 'Limit Analysis (kN)'
t_cols_exp = 'Experimental (kN)'

# Extract 
x_r = data[f_cols[0]].values.tolist()
x_t = data[f_cols[1]].values.tolist()
y_la = data[t_cols_limit].values.tolist()
y_exp = data[t_cols_exp].values.tolist()

# new data
n_d_rl = 0.25 # r/L
n_d_tl = 0.09 # t/L
n_d_γM= 22.7  # γM

# coefficients r/L vs Limit Analysis
b0_r_la, b1_r_la = coeffs(x_r, y_la)
r2_r_la, rmse_r_la, pred_r_la, y_pred_r_la = linear_regression(x_r, y_la, b0_r_la, b1_r_la, n_d_rl, 'r/L (m)', 'Limit Analysis (kN)', 'r/L vs Limit Analysis')

# coefficients t/L vs Limit Analysis
b0_t_la, b1_t_la = coeffs(x_t, y_la)
r2_t_la, rmse_t_la, pred_t_la, y_pred_t_la = linear_regression(x_t, y_la, b0_t_la, b1_t_la, n_d_tl, 't/L', 'Limit Analysis (kN)', 't/L vs Limit Analysis')

# coefficients r/L vs Experimental
b0_r_exp, b1_r_exp = coeffs(x_r, y_exp)
r2_r_exp, rmse_r_exp, pred_r_exp, y_pred_r_exp = linear_regression(x_r, y_exp, b0_r_exp, b1_r_exp, n_d_rl, 'r/L (m)', 'Experimental (kN)', 'r/L vs Experimental')

# coefficients t/L vs Experimental
b0_t_exp, b1_t_exp = coeffs(x_t, y_exp)
r2_t_exp, rmse_t_exp, pred_t_exp, y_pred_t_exp = linear_regression(x_t, y_exp, b0_t_exp, b1_t_exp, n_d_tl, 't/L', 'Experimental (kN)', 't/L vs Experimental')

#  combined data
X = np.column_stack((x_r, x_t))
n_d_comb = [n_d_rl, n_d_tl]

# combined Limit Analysis
r2_comb_la, rmse_comb_la, pred_comb_la, y_pred_comb_la = combined_reg(X, y_la, n_d_comb, 'Limit Analysis')

# combined  Experimental
r2_comb_exp, rmse_comb_exp, pred_comb_exp, y_pred_comb_exp = combined_reg(X, y_exp, n_d_comb, 'Experimental')

# Plotting
plt.figure(figsize=(12, 10))

# Plot r/L vs Limit Analysis
plt.subplot(2, 2, 1)
plot_regression(x_r, y_la, b0_r_la, b1_r_la, n_d_rl, 'r/L (m)', 'Limit Analysis (kN)', 'r/L vs Limit Analysis')

# Plot r/L vs Experimental
plt.subplot(2, 2, 2)
plot_regression(x_r, y_exp, b0_r_exp, b1_r_exp, n_d_rl, 'r/L (m)', 'Experimental (kN)', 'r/L vs Experimental')

# Plot t/L vs Limit Analysis
plt.subplot(2, 2, 3)
plot_regression(x_t, y_la, b0_t_la, b1_t_la, n_d_tl, 't/L', 'Limit Analysis (kN)', 't/L vs Limit Analysis')

# Plot t/L vs Experimental
plt.subplot(2, 2, 4)
plot_regression(x_t, y_exp, b0_t_exp, b1_t_exp, n_d_tl, 't/L', 'Experimental (kN)', 't/L vs Experimental')

plt.tight_layout()
plt.show()
