from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time

def evaluate_model(y_true, y_pred):
    # Manual RMSE to avoid 'squared' compatibility issues
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred)
    }

def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start
