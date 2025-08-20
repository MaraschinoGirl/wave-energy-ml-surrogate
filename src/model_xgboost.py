# src/model_xgboost.py
from xgboost import XGBRegressor

def train_xgboost_model(X, y):
    model = XGBRegressor()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

