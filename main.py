# main.py

import sklearn
print("✅ Scikit-learn version in use:", sklearn.__version__)

# ------------------------------------------------------------
# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_and_merge
from src.preprocessing import (
    clean_data, normalize_columns, split_data,
    add_lag_feature, add_location_embedding, add_pca_features
)
from src.features_layout import add_layout_features
from src.data_preparation import prepare_lstm_data
from src.model_xgboost import train_xgboost_model, predict
from src.model_lstm import train_lstm_model, predict_qw
from src.evaluate import evaluate_model, time_function
from src.runtime_analysis import measure_runtime, dummy_evolutionary_algorithm
from src import visuals

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from copy import deepcopy
import logging
import os

# ------------------------------------------------------------
#  Load and clean datasets
perth_data = load_and_merge(config.PERTH_FILES, 'Perth')
sydney_data = load_and_merge(config.SYDNEY_FILES, 'Sydney')

perth_data = clean_data(perth_data)
sydney_data = clean_data(sydney_data)

# ------------------------------------------------------------
#  Normalize power-related columns
# Define the target and power-related columns
power_cols = [col for col in perth_data.columns if 'Power' in col or col == 'Total_Power']
qw_col = ['qW']

# Normalize qW separately and keep the scaler for LSTM & cross-location consistency
perth_data, qw_scaler = normalize_columns(perth_data, qw_col, return_scaler=True)

# Normalize other power-related cols (like Total_Power) with a new scaler
perth_data = normalize_columns(perth_data, power_cols)

# Apply Perth's qW scaler to Sydney qW ONLY
sydney_data = normalize_columns(sydney_data, qw_col, scaler=qw_scaler, fit=False)

# Normalize Sydney's remaining power columns with separate scaler
sydney_data = normalize_columns(sydney_data, power_cols)
#To denormalize later: y_pred_actual = qw_scaler.inverse_transform(predicted_qw.reshape(-1, 1)).flatten()



# ------------------------------------------------------------
#  Add features
perth_data = add_layout_features(perth_data)
sydney_data = add_layout_features(sydney_data)

perth_data = add_location_embedding(perth_data)
sydney_data = add_location_embedding(sydney_data)

perth_data = add_pca_features(perth_data)
sydney_data = add_pca_features(sydney_data)

# ------------------------------------------------------------
# XGBoost (RQ1)
X_train, X_test, y_train, y_test = split_data(perth_data)
model, train_time = time_function(train_xgboost_model, X_train, y_train)
y_pred_test, _ = time_function(predict, model, X_test)
print("Test Set Metrics (Perth):", evaluate_model(y_test, y_pred_test))

X_sydney = sydney_data.drop(columns=['Total_Power'])
y_sydney = sydney_data['Total_Power']
y_pred_sydney, _ = time_function(predict, model, X_sydney)
print("Generalization Metrics (Sydney):", evaluate_model(y_sydney, y_pred_sydney))

# ------------------------------------------------------------
# LSTM (RQ2)
# Get top features based on importance
def get_top_features(model, feature_names, top_n=15):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    return [feature_names[i] for i in sorted_idx[:top_n]]

top_lstm_features = get_top_features(model, X_train.columns, top_n=15)

# Prepare LSTM data using only these features
X_lstm, y_lstm = prepare_lstm_data(perth_data, top_features=top_lstm_features)

print(" Filtered X_lstm shape:", X_lstm.shape)
print("Top LSTM input features based on XGBoost:")
for f in top_lstm_features:
    print(f)

sns.histplot(y_lstm, bins=50, kde=True)
plt.title("Distribution of qW (Perth Training Labels)")
plt.tight_layout()
plt.savefig("figures/qw_train_distribution.png")
plt.close()

print(" X_lstm shape:", X_lstm.shape)
print(" y_lstm sample:", y_lstm[:10])
print(" One input sample (X_lstm[0]):\n", X_lstm[0])

lstm_model, history = train_lstm_model(X_lstm, y_lstm)
print(f" Final LSTM training loss: {history.history['loss'][-1]:.5f}")
print(f" Final LSTM validation loss: {history.history['val_loss'][-1]:.5f}")

X_sydney_lstm, y_sydney_qw = prepare_lstm_data(sydney_data, top_features=top_lstm_features)

predicted_qw = predict_qw(lstm_model, X_sydney_lstm)

print("\n LSTM Performance (Sydney qW):")
print("MAE:", evaluate_model(y_sydney_qw, predicted_qw)['MAE'])
print("RMSE:", evaluate_model(y_sydney_qw, predicted_qw)['RMSE'])
print("R2:", evaluate_model(y_sydney_qw, predicted_qw)['R2'])

# ------------------------------------------------------------
#  RQ2: XGBoost with inferred qW
predicted_qw = predict_qw(lstm_model, X_sydney_lstm)

# Denormalize predicted qW
predicted_qw_actual = qw_scaler.inverse_transform(predicted_qw.reshape(-1, 1)).flatten()

sydney_data_inferred = sydney_data.copy()
sequence_length = 10
sydney_data_inferred.loc[sydney_data_inferred.index[sequence_length:], 'qW'] = predicted_qw_actual


X_sydney_inferred = sydney_data_inferred.drop(columns=['Total_Power'])
y_sydney_true = sydney_data_inferred['Total_Power']
y_pred_sydney_inferred, _ = time_function(predict, model, X_sydney_inferred)
print("\n Generalization (Sydney) with LSTM-inferred qW:")
print(evaluate_model(y_sydney_true, y_pred_sydney_inferred))

# ------------------------------------------------------------
#  Baselines
print("\n Baseline Model Comparisons (Perth Test Set):")
for name, model_cls in [("Linear Regression", LinearRegression), ("Random Forest", RandomForestRegressor)]:
    print(f"\n {name}:")
    baseline_model = model_cls()
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)
    print(evaluate_model(y_test, y_pred))

# ------------------------------------------------------------
#  RQ4: Runtime Comparison
print("\n Benchmarking Runtime...")
_, xgb_time = measure_runtime(train_xgboost_model, X_train, y_train)
print(f"XGBoost Training Time: {xgb_time:.4f} seconds")

_, lstm_time = measure_runtime(train_lstm_model, X_lstm, y_lstm)
print(f"LSTM Training Time: {lstm_time:.4f} seconds")

def rf_train(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

_, rf_time = measure_runtime(rf_train, X_train, y_train)
print(f"Random Forest Training Time: {rf_time:.4f} seconds")

dummy_layout = np.random.rand(100, 2)
ea_time = dummy_evolutionary_algorithm(dummy_layout)
print(f"Dummy EA Time (1000 iterations): {ea_time:.4f} seconds")

# ------------------------------------------------------------
# RQ3: Transfer Learning
print("\n RQ3: Transfer Learning - Fine-Tuning XGBoost on small Sydney subset...")
X_ft, X_test_syd, y_ft, y_test_syd = train_test_split(X_sydney, y_sydney, test_size=0.8, random_state=42)
model_ft = deepcopy(model)
model_ft.fit(X_ft, y_ft)
y_pred_ft = model_ft.predict(X_test_syd)
print(" Fine-tuned XGBoost on Sydney (20% of data):")
print(evaluate_model(y_test_syd, y_pred_ft))

# ------------------------------------------------------------
#  Visualizations
visuals.plot_qw_inference(
    qw_scaler.inverse_transform(y_sydney_qw.reshape(-1, 1)).flatten(),
    qw_scaler.inverse_transform(predicted_qw.reshape(-1, 1)).flatten()
)

visuals.plot_predicted_vs_true(y_sydney, y_pred_sydney)
visuals.plot_feature_importance(model, X_train.columns)
visuals.plot_layout_hexbin(perth_data)
visuals.plot_lstm_loss(history)
visuals.plot_location_comparison(
    evaluate_model(y_test, y_pred_test),
    evaluate_model(y_sydney, y_pred_sydney)
)

visuals.plot_runtime_comparison({
    "XGBoost": xgb_time,
    "Random Forest": rf_time,
    "LSTM": lstm_time,
    "Dummy EA (Simulated)": ea_time
})

# ------------------------------------------------------------
#  Delta Table
real_metrics = evaluate_model(y_sydney, y_pred_sydney)
inferred_metrics = evaluate_model(y_sydney_true, y_pred_sydney_inferred)
delta_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Real qW": [real_metrics["MAE"], real_metrics["RMSE"], real_metrics["R2"]],
    "Inferred qW": [inferred_metrics["MAE"], inferred_metrics["RMSE"], inferred_metrics["R2"]],
})
delta_df["Δ (Abs)"] = delta_df["Inferred qW"] - delta_df["Real qW"]
delta_df["Δ (%)"] = ((delta_df["Δ (Abs)"] / delta_df["Real qW"]) * 100).round(2)
print("\n XGBoost Performance Delta (Real vs. Inferred qW):")
print(delta_df.to_string(index=False))
delta_df.to_csv("figures/xgboost_delta_metrics.csv", index=False)

# ------------------------------------------------------------
#  LSTM on Perth (for sanity)
train_preds = predict_qw(lstm_model, X_lstm)
print("\n LSTM Performance (on Perth Training Set):")
print(evaluate_model(y_lstm, train_preds))
print("Train Pred Stats:")
print(f"Min: {train_preds.min():.4f}, Max: {train_preds.max():.4f}, Std: {train_preds.std():.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_lstm[:200], label='Actual qW')
plt.plot(predict_qw(lstm_model, X_lstm[:200]), label='Predicted qW')
plt.legend()
plt.title("LSTM Prediction vs. Actual on Perth")
plt.tight_layout()
plt.savefig("figures/lstm_qw_pred_vs_true_perth.png")
plt.close()

# ------------------------------------------------------------
#  Logging and diagnostics
os.makedirs("figures", exist_ok=True)
logging.basicConfig(filename='runlog.txt', level=logging.INFO)
logging.info("Run complete.")
