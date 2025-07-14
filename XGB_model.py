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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import xgboost as xgb

# Step 1: Preprocessing the first dataset and building the model
file_path = r".\Sample_data.xlsx"

# Load 'H2O2' sheet
data = pd.read_excel(file_path, sheet_name='H2O2', header=0)
data = data[data.iloc[:, 1] == 'MOF']  # Filter rows where the 2nd column is 'MOF'
data.head()

# Define input and target variables
X = data.iloc[:, 2:9]  # Columns 3 to 9 (input variables)
y = data.iloc[:, 9]    # Column 10 (target variable)

# Preprocessing: One-hot encode the first column and apply MinMax scaling to the rest
ohe = OneHotEncoder(sparse=False)
scaler = MinMaxScaler()

X_cat = ohe.fit_transform(X.iloc[:, 0].values.reshape(-1, 1))
X_num = scaler.fit_transform(X.iloc[:, 1:])

X_processed = np.hstack((X_cat, X_num))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Define the objective function for Bayesian optimization
def objective_1(params):
    model = xgb.XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        min_child_weight=int(params['min_child_weight']),
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    return {'loss': loss, 'status': STATUS_OK}

# Define the hyperparameter search space
param_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
}

# Run Bayesian optimization
trials = Trials()
best_params = fmin(
    fn=objective_1,
    space=param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials,
    rstate=np.random.default_rng(42)
)

# Train model using the optimized hyperparameters
optimized_xgb = xgb.XGBRegressor(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']),
    learning_rate=float(best_params['learning_rate']),
    min_child_weight=int(best_params['min_child_weight']),
    random_state=42,
    verbosity=0
)
optimized_xgb.fit(X_train, y_train)

# Define evaluation function
def evaluate_model(model, X, y, label):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"{label} - R²: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return y_pred

# Evaluate model on training and test data
y_train_pred = evaluate_model(optimized_xgb, X_train, y_train, "Train")
y_test_pred = evaluate_model(optimized_xgb, X_test, y_test, "Test")

# Visualize prediction results
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, label='Train data', alpha=0.6, color='blue')
plt.scatter(y_test, y_test_pred, label='Test data', alpha=0.6, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel("Observed values", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.ylabel("Predicted values", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.xticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.yticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.legend(fontsize=16, loc='best', prop={'family': 'Times New Roman', 'weight': 'bold'})
plt.title("Observed Performace vs Predicted Performace", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.show()

######################################################################################################
# Step 2: Preprocessing the second dataset and retraining the model
performance_data = pd.read_excel(file_path, sheet_name='Performance', header=0)
performance_data = performance_data[performance_data.iloc[:, 1] == 'MOF']

# Remove duplicates and apply filtering
performance_data = performance_data.drop_duplicates(subset=performance_data.columns[2:10])
performance_data = performance_data[performance_data.iloc[:, 5] != 0]
performance_data = performance_data[~performance_data.iloc[:, 7].isin([3.5, 10.5])]
performance_data = performance_data[~performance_data.iloc[:, 2].isin(['MB', 'AB'])]

# Define input and target variables
X_perf = performance_data.iloc[:, 2:9]
y_perf = performance_data.iloc[:, 9]

# Apply preprocessing
X_cat_perf = ohe.transform(X_perf.iloc[:, 0].values.reshape(-1, 1))
X_num_perf = scaler.transform(X_perf.iloc[:, 1:])
X_perf_processed = np.hstack((X_cat_perf, X_num_perf))

# Predict using the first model
y_perf_pred = optimized_xgb.predict(X_perf_processed)
scaler_pred = MinMaxScaler()
y_perf_pred_scaled = scaler_pred.fit_transform(y_perf_pred.reshape(-1, 1))

# Append scaled predictions to input features
X_perf_with_pred = np.hstack((X_perf_processed, y_perf_pred_scaled))

# Train-test split for the new model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_perf_with_pred, y_perf, test_size=0.3, random_state=42)

# Define the objective function for Bayesian optimization (second stage)
def objective_2(params):
    model = xgb.XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=float(params['learning_rate']),
        min_child_weight=int(params['min_child_weight']),
        random_state=42,
        verbosity=0
    )
    model.fit(X_train2, y_train2)
    y_pred2 = model.predict(X_test2)
    loss = mean_squared_error(y_test2, y_pred2)
    return {'loss': loss, 'status': STATUS_OK}

# Run optimization for second model
trials2 = Trials()
best_params2 = fmin(
    fn=objective_2,
    space=param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials2,
    rstate=np.random.default_rng(42)
)

# Train final model with optimized parameters
optimized_xgb2 = xgb.XGBRegressor(
    n_estimators=int(best_params2['n_estimators']),
    max_depth=int(best_params2['max_depth']),
    learning_rate=float(best_params2['learning_rate']),
    min_child_weight=int(best_params2['min_child_weight']),
    random_state=42,
    verbosity=0
)
optimized_xgb2.fit(X_train2, y_train2)

# Evaluate and visualize final model
y_train2_pred = evaluate_model(optimized_xgb2, X_train2, y_train2, "Train (New)")
y_test2_pred = evaluate_model(optimized_xgb2, X_test2, y_test2, "Test (New)")

plt.figure(figsize=(8, 6))
plt.scatter(y_train2, y_train2_pred, label='Train data', alpha=0.6, color='blue')
plt.scatter(y_test2, y_test2_pred, label='Test data', alpha=0.6, color='red')
plt.plot([y_perf.min(), y_perf.max()], [y_perf.min(), y_perf.max()], '--k')
plt.xlabel("Observed values", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.ylabel("Predicted values", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.xticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.yticks(fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.legend(fontsize=16, loc='best', prop={'family': 'Times New Roman', 'weight': 'bold'})
plt.title("Observed Performace vs Predicted Performace", fontname="Times New Roman", fontsize=14, fontweight='bold')
plt.show()