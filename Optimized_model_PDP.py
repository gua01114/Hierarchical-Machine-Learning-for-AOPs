######################## 00. Clear ##################################
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from scipy.signal import savgol_filter

# Step 1: Preprocess the first dataset and load the model
file_path = r".\Sample_data.xlsx"
catboost_first_model_path = r".\CB_1_MOF.pkl"
catboost_second_model_path = r".\CB_2_MOF.pkl"

# Read the 'H2O2' sheet
data = pd.read_excel(file_path, sheet_name='H2O2', header=0)
data = data[data.iloc[:, 1] == 'MOF']  # Filter rows where the second column is 'MOF'

# Define input and target variables
X = data.iloc[:, 2:9]  # Columns 3 to 9 (input variables)
y = data.iloc[:, 9]    # Column 10 (target variable)

# Preprocessing: one-hot encode the first column, min-max scale the rest
ohe = OneHotEncoder(sparse=False)
scaler = MinMaxScaler()

X_cat = ohe.fit_transform(X.iloc[:, 0].values.reshape(-1, 1))
X_num = scaler.fit_transform(X.iloc[:, 1:])

X_processed = np.hstack((X_cat, X_num))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Load the optimized CatBoost model (first stage)
optimized_cb = joblib.load(catboost_first_model_path)

# Define evaluation function
def evaluate_model(model, X, y, label):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"{label} - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return y_pred

# Evaluate on training and testing data
y_train_pred = evaluate_model(optimized_cb, X_train, y_train, "Train")
y_test_pred = evaluate_model(optimized_cb, X_test, y_test, "Test")

######################################################################################################
# Step 2: Preprocess the second dataset and reload the model
performance_data = pd.read_excel(file_path, sheet_name='Performance', header=0)
performance_data = performance_data[performance_data.iloc[:, 1] == 'MOF']

# Remove duplicates
performance_data = performance_data.drop_duplicates(subset=performance_data.columns[2:10])
performance_data = performance_data[performance_data.iloc[:, 5] != 0]
performance_data = performance_data[~performance_data.iloc[:, 7].isin([3.5, 10.5])]
performance_data = performance_data[~performance_data.iloc[:, 2].isin(['MB', 'AB'])]

# Define input and target variables
X_perf = performance_data.iloc[:, 2:9]
y_perf = performance_data.iloc[:, 9]

# Apply same preprocessing
X_cat_perf = ohe.transform(X_perf.iloc[:, 0].values.reshape(-1, 1))
X_num_perf = scaler.transform(X_perf.iloc[:, 1:])
X_perf_processed = np.hstack((X_cat_perf, X_num_perf))

# Predict using the first model
y_perf_pred = optimized_cb.predict(X_perf_processed)
scaler_pred = MinMaxScaler()
y_perf_pred_scaled = scaler_pred.fit_transform(y_perf_pred.reshape(-1, 1))

# Add the scaled prediction as a new feature
X_perf_with_pred = np.hstack((X_perf_processed, y_perf_pred_scaled))

# Train-test split and load second stage model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_perf_with_pred, y_perf, test_size=0.3, random_state=42)
optimized_cb2 = joblib.load(catboost_second_model_path)

# Evaluate and visualize
y_train2_pred = evaluate_model(optimized_cb2, X_train2, y_train2, "Train (New)")
y_test2_pred = evaluate_model(optimized_cb2, X_test2, y_test2, "Test (New)")


# PDP calculation function (with inverse scaling)
def calculate_pdp_with_original_scale(model, X, feature_index, feature_range, original_feature_values):
    pdp_values = []
    for value in feature_range:
        X_temp = X.copy()
        X_temp[:, feature_index] = value
        preds = model.predict(X_temp)
        pdp_values.append(np.mean(preds))

    # Inverse transform to original scale
    scaler = MinMaxScaler()
    scaler.fit(original_feature_values.reshape(-1, 1))
    original_values = scaler.inverse_transform(feature_range.reshape(-1, 1)).flatten()

    return original_values, pdp_values

# Extract original (unscaled) feature values
original_feature_values = y_perf_pred.reshape(-1, 1)

# Compute PDP
feature_index = 9  # Index of feature to analyze
feature_range = np.linspace(
    X_perf_with_pred[:, feature_index].min(),
    X_perf_with_pred[:, feature_index].max(),
    100
)

original_feature_values, pdp_values = calculate_pdp_with_original_scale(
    optimized_cb2, X_perf_with_pred, feature_index, feature_range, original_feature_values
)

# Smooth PDP curve using Savitzky-Golay filter
smoothed_pdp = savgol_filter(pdp_values, window_length=11, polyorder=3)

# Compute first and second derivatives
first_derivative = np.gradient(smoothed_pdp, original_feature_values)
second_derivative = np.gradient(first_derivative, original_feature_values)

# Identify regions where second derivative is negative and small in magnitude
negative_second_deriv = second_derivative[second_derivative < 0]
abs_neg_sd = np.abs(first_derivative)
threshold = np.percentile(abs_neg_sd, 10)

# Find indices satisfying the conditions
target_indices = np.where((second_derivative < 0) & (np.abs(second_derivative) <= threshold))[0]

# Step 4: Final plot using original PDP
plt.figure(figsize=(4.5, 4.5))
plt.plot(original_feature_values, smoothed_pdp, label='Smoothed PDP', color='dimgray', linewidth=2)
plt.plot(original_feature_values, pdp_values, label='Original PDP', color='blue', linewidth=2)

# Mark key turning point
x_val = original_feature_values[target_indices[1]]
y_val = pdp_values[target_indices[1]]
plt.axvline(x=x_val, color='black', linestyle='--', linewidth=2)
plt.plot(x_val, y_val, 'o', color='black')

plt.xlabel("H2O2 production (ppm)", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.ylabel("Mean predicted degradation efficiency", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.xticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.yticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
