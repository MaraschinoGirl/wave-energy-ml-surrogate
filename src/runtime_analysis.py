# src/runtime_analysis.py

import time
from src.model_xgboost import train_xgboost_model
from src.model_lstm import train_lstm_model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def measure_runtime(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def dummy_evolutionary_algorithm(layout_input, iterations=1000):
    # Simulate an EA taking longer with more iterations
    start = time.perf_counter()
    for _ in range(iterations):
        np.random.shuffle(layout_input)
        _ = np.mean(layout_input)
    end = time.perf_counter()
    return end - start
