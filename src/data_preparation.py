# src/data_preparation.py
import pandas as pd
import numpy as np
def prepare_lstm_data(df, target='qW', sequence_length=10, top_features=None):
    df = df.copy()

    # If top_features is given, select only those (including target column)
    if top_features is not None:
        missing = [col for col in top_features if col not in df.columns]
        if missing:
            raise KeyError(f"These top_features are missing from df: {missing}")
        features = df[top_features].values
    else:
        # Drop irrelevant power cols if not using top features
        power_cols = [col for col in df.columns if "Power" in col and col != target]
        df = df.drop(columns=power_cols + ['Total_Power'])
        features = df.drop(columns=[target]).values

    labels = df[target].values

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(features[i - sequence_length:i])
        y.append(labels[i])

    return np.array(X), np.array(y)
