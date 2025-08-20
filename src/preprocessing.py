# src/preprocessing.py
import numpy as np

from sklearn.model_selection import train_test_split

def split_data(df, target='Total_Power'):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# src/preprocessing.py
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df



from sklearn.preprocessing import MinMaxScaler

def normalize_columns(df, cols, scaler=None, fit=True, return_scaler=False):
    if scaler is None:
        scaler = MinMaxScaler()
    if fit:
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    if return_scaler:
        return df, scaler
    return df


# src/preprocessing.py
def add_lag_feature(df, col, lag=1):
    df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df.dropna()

# src/preprocessing.py
def add_location_embedding(df):
    import numpy as np
    import pandas as pd

    x_cols = sorted([col for col in df.columns if col.startswith("X")])
    y_cols = sorted([col for col in df.columns if col.startswith("Y")])

    # Calculate all distances at once using a dictionary
    distances = {
        f'dist_{x}_{y}': np.sqrt(df[x]**2 + df[y]**2)
        for x, y in zip(x_cols, y_cols)
    }

    # Combine into a temporary DataFrame
    dist_df = pd.DataFrame(distances)

    # Append + compute mean distance
    df = pd.concat([df, dist_df], axis=1)
    df['MeanWEC_DistanceFromOrigin'] = dist_df.mean(axis=1)

    # Drop intermediate columns
    df = df.drop(columns=dist_df.columns)

    return df


from sklearn.decomposition import PCA

def add_pca_features(df, n_components=3):
    coord_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df[coord_cols])
    for i in range(n_components):
        df[f'pca_coord_{i+1}'] = reduced[:, i]
    return df

