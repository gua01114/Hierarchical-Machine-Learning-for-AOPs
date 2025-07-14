######################## 00. Clear ##################################
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import warnings
warnings.filterwarnings(action='ignore')

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import Pool
from scipy.stats import ttest_ind

# Set file paths
file_path = r".\Sample_data"
random_states = 42

# Load the first CatBoost model
catboost_first_model_path = r".\CB_1_MOF.pkl"
catboost_first_model = joblib.load(catboost_first_model_path)
# Load the second CatBoost model
catboost_second_model_path = r".\CB_2_MOF.pkl"
catboost_second_model = joblib.load(catboost_second_model_path)

# Load and preprocess the first dataset
data = pd.read_excel(file_path, sheet_name='H2O2', header=0)
data = data[data.iloc[:, 1] == 'MOF']
X = data.iloc[:, 2:9]
y = data.iloc[:, 9]

ohe = OneHotEncoder(sparse=False)
scaler = MinMaxScaler()
X_cat = ohe.fit_transform(X.iloc[:, 0].values.reshape(-1, 1))
X_num = scaler.fit_transform(X.iloc[:, 1:])
X_processed = np.hstack((X_cat, X_num))

# Split the first dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=random_states)

# Load and preprocess the second dataset
performance_data = pd.read_excel(file_path, sheet_name='Performance', header=0)
performance_data = performance_data[performance_data.iloc[:, 1] == 'MOF']
performance_data = performance_data.drop_duplicates(subset=performance_data.columns[2:10])
performance_data = performance_data[performance_data.iloc[:, 5] != 0]
performance_data = performance_data[~performance_data.iloc[:, 7].isin([3.5, 10.5])]
performance_data = performance_data[~performance_data.iloc[:, 2].isin(['MB', 'AB'])]

X_perf = performance_data.iloc[:, 2:9]
y_perf = performance_data.iloc[:, 9]

X_cat_perf = ohe.transform(X_perf.iloc[:, 0].values.reshape(-1, 1))
X_num_perf = scaler.transform(X_perf.iloc[:, 1:])
X_perf_processed = np.hstack((X_cat_perf, X_num_perf))

# Generate predictions from the first model
y_perf_pred = catboost_first_model.predict(X_perf_processed)
scaler_pred = MinMaxScaler()
y_perf_pred_scaled = scaler_pred.fit_transform(y_perf_pred.reshape(-1, 1))

# Add scaled predictions to the input features
X_perf_with_pred = np.hstack((X_perf_processed, y_perf_pred_scaled))

# Split the second dataset
X_perf_train_with_pred, X_perf_test_with_pred, y_perf_train_split, y_perf_test_split = train_test_split(
    X_perf_with_pred, y_perf, test_size=0.3, random_state=random_states
)

# Prepare Pool objects for SHAP
pool_first_all = Pool(X_processed, y)
shap_values_first_all = catboost_first_model.get_feature_importance(pool_first_all, type="ShapValues")

pool_second_all = Pool(X_perf_with_pred, y_perf)
shap_values_second_all = catboost_second_model.get_feature_importance(pool_second_all, type="ShapValues")

# Function to save SHAP values to dataframe
def save_shap_values(shap_values, feature_names):
    shap_df = pd.DataFrame(shap_values[:, :-1], columns=feature_names)
    return shap_df

# Save SHAP values for the first model
SHAP_1st = save_shap_values(
    shap_values=shap_values_first_all,
    feature_names=ohe.get_feature_names_out().tolist() + data.columns[3:9].tolist()
)

# Save SHAP values for the second model
SHAP_2nd= save_shap_values(
    shap_values=shap_values_second_all,
    feature_names=ohe.get_feature_names_out().tolist() + data.columns[3:9].tolist() + ["First Model Prediction"]
)

# Dictionary for model names and plot titles
shap_dataframes = {
    "First": SHAP_1st,
    "Second": SHAP_2nd
}

titles = {
    "First": "H₂O₂ production prediction model",
    "Second": "Performance prediction model"
}

# Function to process data and generate boxplots with t-tests
def process_and_plot(data, title):
    # Replace column name H2O2 with formatted H₂O₂ for display
    data.columns = [col.replace("H2O2", "H$_2$O$_2$") for col in data.columns]

    # Filter features: remove first 3 and 9th columns; use absolute SHAP values
    filtered_data = data.iloc[:, list(range(3, 8)) + list(range(9, data.shape[1]))]
    filtered_data = filtered_data.abs()

    # Compute column means and sort by descending order
    means = filtered_data.mean(axis=0)
    sorted_indices = means.sort_values(ascending=False).index
    filtered_data = filtered_data[sorted_indices]
    sorted_means = means[sorted_indices]

    # Perform pairwise t-tests between adjacent features
    t_test_results = []
    significant_pairs = []
    columns = filtered_data.columns

    for i in range(len(columns) - 1):
        feature_1 = columns[i]
        feature_2 = columns[i + 1]
        data_1 = filtered_data[feature_1]
        data_2 = filtered_data[feature_2]

        t_stat, p_value = ttest_ind(data_1, data_2, equal_var=False)

        t_test_results.append({
            "Feature 1": feature_1,
            "Feature 2": feature_2,
            "Mean 1": sorted_means[i],
            "Mean 2": sorted_means[i + 1],
            "T-Statistic": t_stat,
            "P-Value": p_value
        })

        if p_value < 0.05:
            significant_pairs.append((feature_1, feature_2, p_value))

    t_test_df = pd.DataFrame(t_test_results)

    # Print t-test results
    print(f"T-Test Results for {title}:")
    print(t_test_df)

    # Create horizontal boxplot
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        filtered_data.values,
        vert=False,
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=1.5, color='red'),
        flierprops=dict(marker='o', markersize=5, linestyle='none', markeredgecolor='black')
    )

    # Set axis labels and title
    plt.yticks(
        ticks=range(1, len(filtered_data.columns) + 1),
        labels=filtered_data.columns,
        fontname='Times New Roman',
        fontsize=14,
        fontweight='bold'
    )
    plt.xticks(fontname='Times New Roman', fontsize=14, fontweight='bold')
    plt.xlabel('Mean absolute SHAP value', fontname='Times New Roman', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontname='Times New Roman', fontsize=14, fontweight='bold')
    plt.title(title, fontname='Times New Roman', fontsize=14, fontweight='bold')

    # Add mean values as text annotations
    x_offset = filtered_data.max().max() * 1.07
    text_positions = []
    for i, mean_value in enumerate(sorted_means, start=1):
        text = plt.text(
            x=x_offset,
            y=i,
            s=f"Mean: {mean_value:.2f}",
            fontname='Times New Roman',
            fontsize=14,
            fontweight='bold',
            va='center'
        )
        text_positions.append(text.get_position())

    # Mark statistically significant differences (p < 0.05) with an asterisk
    for (feature_1, feature_2, p_value) in significant_pairs:
        idx1 = filtered_data.columns.get_loc(feature_1) + 1
        idx2 = filtered_data.columns.get_loc(feature_2) + 1
        mid_y = (idx1 + idx2) / 2
        mid_x = (text_positions[idx1 - 1][0] + text_positions[idx2 - 1][0]) / 2
        plt.text(
            x=mid_x,
            y=mid_y,
            s="*",
            fontsize=14,
            fontweight='bold',
            color='blue',
            va='center',
            ha='center'
        )

    # Reverse y-axis so highest SHAP values appear at top
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Apply processing and plotting for both models
for key in shap_dataframes:
    process_and_plot(shap_dataframes[key], titles[key])
