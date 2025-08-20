# config.py

# File paths to input CSV datasets
PERTH_FILES = [
    "data/WEC_Perth_49.csv",
    "data/WEC_Perth_100.csv"
]

SYDNEY_FILES = [
    "data/WEC_Sydney_49.csv",
    "data/WEC_Sydney_100.csv"
]


# LSTM-specific settings
LSTM_SEQUENCE_LENGTH = 10

# Feature settings (if needed globally)
PCA_COMPONENTS = 3
