# src/data_loader.py
import pandas as pd

def load_and_merge(files, label):
    df_list = []
    for file in files:
        count = int(file.split('_')[-1].replace('.csv', ''))
        df = pd.read_csv(file)
        df['WEC_Count'] = count
        df['Site'] = label
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

from sklearn.preprocessing import LabelEncoder

def load_and_merge(files, label):
    df_list = []
    for file in files:
        count = int(file.split('_')[-1].replace('.csv', ''))
        df = pd.read_csv(file)
        df['WEC_Count'] = count
        df['Site'] = label
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # Convert Site to categorical (int)
    le = LabelEncoder()
    merged_df['Site'] = le.fit_transform(merged_df['Site'])

    return merged_df
