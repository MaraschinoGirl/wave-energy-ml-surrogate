import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde


os.makedirs("figures", exist_ok=True)


def plot_qw_inference(y_true, y_pred, save_path="figures/qw_inference_sydney.png"):
    plt.figure(figsize=(10, 6))


    if len(y_pred) != len(y_true):
        min_len = min(len(y_pred), len(y_true))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]


    zoom_len = 1000
    plt.plot(y_true[:zoom_len], label="Actual qW", alpha=0.7)
    plt.plot(y_pred[:zoom_len], label="Predicted qW", alpha=0.7)

    plt.xlabel("Sample Index (Time Ordered)")
    plt.ylabel("qW (Normalized)")
    plt.title("LSTM-Inferred qW vs. Actual qW (Sydney)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predicted_vs_true(y_true, y_pred, save_path="figures/pred_vs_true_power_sydney.png"):
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
    plt.xlabel("Actual Power Output")
    plt.ylabel("Predicted Power Output")
    plt.title("Predicted vs Actual Power Output (Sydney)")
    plt.text(0.05, 0.95, f"$R^2$: {r2:.3f}", transform=plt.gca().transAxes, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path="figures/xgboost_feature_importance.png"):
    importances = model.feature_importances_
    importance_df = (
        np.array(list(zip(feature_names, importances)), dtype=[('Feature', 'U50'), ('Importance', 'f4')])
    )
    sorted_df = np.sort(importance_df, order='Importance')[::-1][:15]

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x=sorted_df['Importance'],
        y=sorted_df['Feature'],
        palette='viridis',
        legend=False
    )
    # Rotate long labels for better readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)


    for label in ax.get_yticklabels():
        if len(label.get_text()) > 20:
            label.set_rotation(90)
            label.set_ha('center')

    plt.title("Top 15 XGBoost Feature Importances", fontsize=14)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)

    # Add importance score labels to bars
    for i, v in enumerate(sorted_df['Importance']):

        ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_layout_hexbin(df, save_path="figures/layout_hexbin_density.png"):
    x_cols = sorted([col for col in df.columns if col.startswith("X")])
    y_cols = sorted([col for col in df.columns if col.startswith("Y")])
    x_vals, y_vals = [], []

    for x_col, y_col in zip(x_cols, y_cols):
        x_vals.extend(df[x_col])
        y_vals.extend(df[y_col])

    # Limit max points for performance
    if len(x_vals) > 10000:
        x_vals = x_vals[:10000]
        y_vals = y_vals[:10000]

    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(x_vals, y_vals, gridsize=50, cmap="viridis", mincnt=1)
    plt.colorbar(hb, label="WEC Density (counts)")
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.title("WEC Layout Density (Hexbin View)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_lstm_loss(history, save_path="figures/lstm_loss_curves.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_location_comparison(metrics_perth, metrics_sydney, save_path="figures/location_comparison.png"):
    labels = ['MAE', 'RMSE']
    values_perth = [metrics_perth['MAE'], metrics_perth['RMSE']]
    values_sydney = [metrics_sydney['MAE'], metrics_sydney['RMSE']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(x - width/2, values_perth, width, label='Perth')
    bars2 = plt.bar(x + width/2, values_sydney, width, label='Sydney')

    # Add numbers on top of both sets of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.ylabel('Error', fontsize=12)
    plt.title('MAE/RMSE Comparison: Perth vs Sydney', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_runtime_comparison(runtime_dict, save_path="figures/runtime_comparison.png"):
    plt.figure(figsize=(10, 6))
    names = list(runtime_dict.keys())
    times = list(runtime_dict.values())
    bars = plt.bar(names, times, color='skyblue')
    plt.title("Training Time Comparison")
    plt.ylabel("Seconds")
    plt.xticks(rotation=45)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, f"{t:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_wec_density(df, save_path="figures/layout_density.png"):
    x_cols = sorted([col for col in df.columns if col.startswith("X")])
    y_cols = sorted([col for col in df.columns if col.startswith("Y")])
    x_vals, y_vals = [], []

    for x_col, y_col in zip(x_cols, y_cols):
        x_vals.extend(df[x_col])
        y_vals.extend(df[y_col])

    # Reduce point count to prevent freezing
    if len(x_vals) > 5000:
        x_vals = x_vals[:5000]
        y_vals = y_vals[:5000]

    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        x=x_vals, y=y_vals,
        cmap="viridis", fill=True,
        levels=20, thresh=0.01
    )

    plt.title("WEC Layout Density (Contour View)", fontsize=14)
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
