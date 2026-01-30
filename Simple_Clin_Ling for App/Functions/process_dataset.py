import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_single_dataset(df):

    TARGET = "Noun_Acc"

    # Columns to drop if present
    cols_to_drop = [
        'Time_Point', 'Set', 'SubjID', 'Travel solo', 'Diff_WAB_AQ',
        'Baseline', 'Trial', 'Card', 'Agent_Noun', 'Item_Type',
        'WAB_AQ_1', 'WAB_AQ_2', 'Race', 'Sex', 'Ethnicity ',
        'Handed', 'Aphasia_Type_1', 'Alt_Noun'
    ]

    # Columns requiring label encoding
    col_encoders = {
        "Handed": LabelEncoder(),
        "Aphasia_Type_1": LabelEncoder(),
        "Freq_Cond": LabelEncoder(),
        "NA_Cond": LabelEncoder()
    }

    df_clean = df.copy()


    if TARGET not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")

    y_ = df_clean[TARGET].copy()
    df_clean = df_clean.drop(columns=[TARGET])

    drop_cols = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_cols)

    for col, encoder in col_encoders.items():
        if col in df_clean.columns:
            df_clean[col] = encoder.fit_transform(df_clean[col].astype(str))

    if "Lesion_Volume" in df_clean.columns:
        scaler = StandardScaler()
        df_clean["Lesion_Volume"] = scaler.fit_transform(
            df_clean[["Lesion_Volume"]]
        )

    print(f"X shape: {df_clean.shape}")
    print(f"y shape: {y_.shape}")
    print(f"Missing values in X: {df_clean.isna().sum().sum()}")

    return df_clean, y_


def preprocess_single_dataset_simple(df):

    TARGET = "Noun_Acc"

    # WAB_AQ, 


    # Columns to drop if present
    cols_to_drop = [
        'Time_Point', 'Set', 'SubjID', 'Travel solo', 'Diff_WAB_AQ',
        'Baseline', 'Trial', 'Card', 'Agent_Noun', 'Item_Type',
        'WAB_AQ_1', 'WAB_AQ_2', 'Race', 'Sex', 'Ethnicity ',
        'Handed', 'Aphasia_Type_1', 'Alt_Noun',  'CCRSA', 'SS_WAB_Avg', 
        'AVC_WAB_Avg', 'log10_lesion_vol', 'Noun_Freq', 'NA_Cond', 'Morphemes',
        'Noun_NA'
    ]

    # Columns requiring label encoding
    col_encoders = {
        "Handed": LabelEncoder(),
        "Aphasia_Type_1": LabelEncoder(),
        "Freq_Cond": LabelEncoder(),
        "NA_Cond": LabelEncoder()
    }

    df_clean = df.copy()


    if TARGET not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")

    y_ = df_clean[TARGET].copy()
    df_clean = df_clean.drop(columns=[TARGET])

    drop_cols = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_cols)

    for col, encoder in col_encoders.items():
        if col in df_clean.columns:
            df_clean[col] = encoder.fit_transform(df_clean[col].astype(str))

    if "Lesion_Volume" in df_clean.columns:
        scaler = StandardScaler()
        df_clean["Lesion_Volume"] = scaler.fit_transform(
            df_clean[["Lesion_Volume"]]
        )

    print(f"X shape: {df_clean.shape}")
    print(f"y shape: {y_.shape}")
    print(f"Missing values in X: {df_clean.isna().sum().sum()}")

    return df_clean, y_
