# src/model_lstm.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

def train_lstm_model(X, y, epochs=100, batch_size=32):
    if len(X.shape) != 3:
        raise ValueError(f"LSTM input must be 3D. Got shape: {X.shape}")

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss=Huber(delta=1.0),   # Can switch back to 'mse' if needed
        metrics=['mae']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history

def predict_qw(model, X):
    preds = model.predict(X).flatten()
    print(f"Predicted qW → Min: {preds.min():.4f}, Max: {preds.max():.4f}, Std: {preds.std():.4f}")
    return preds

