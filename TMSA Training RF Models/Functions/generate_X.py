import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

def merge_datasets(session_dataset, clinical_dataset, linguistic_dataset, networks_dataset):
    merge_rules = {
        "clin": (session_dataset, clinical_dataset, "SubjID"),
        "ling": (session_dataset, linguistic_dataset, "Agent_Noun"),
        "nets": (session_dataset, networks_dataset, "SubjID"),
        "clin_nets": (pd.merge(session_dataset, clinical_dataset, on="SubjID", how='inner'), networks_dataset, "SubjID"),
        "ling_nets": (pd.merge(session_dataset, linguistic_dataset, on="Agent_Noun", how='inner'), networks_dataset, "SubjID"),
        "clin_ling": (pd.merge(session_dataset, clinical_dataset, on="SubjID", how='inner'), linguistic_dataset, "Agent_Noun"),
        "all3": (pd.merge(pd.merge(session_dataset, clinical_dataset, on="SubjID", how='inner'),
                          linguistic_dataset, on="Agent_Noun", how='inner'), networks_dataset, "SubjID")
    }

    datasets = {}
    for key, (left, right, on_col) in merge_rules.items():
        datasets[key] = pd.merge(left, right, on=on_col, how='inner')
        print(f"{key}: {datasets[key].shape} rows, columns: {list(datasets[key].columns)}")
        
    return datasets

def generate_X(datasets):
    """
    Preprocess a dict of datasets: drop unwanted columns, label encode categorical columns,
    and standardize 'Lesion_Volume' if present.
    Returns a dict of processed DataFrames.
    """

    cols_to_drop = [
        'Time_Point', 'Set', 'SubjID', 'Travel solo', 'Diff_WAB_AQ',
        'Baseline', 'Trial', 'Card', 'Agent_Noun', 'Noun_Acc', 
        'Item_Type', 'WAB_AQ_1', 'WAB_AQ_2', 'Race', 'Sex', 'Ethnicity ',
        'Handed', 'Aphasia_Type_1'
    ]

    col_encoders = {
        "Handed": LabelEncoder(),
        "Aphasia_Type_1": LabelEncoder(),
        "Freq_Cond": LabelEncoder(),
        "NA_Cond": LabelEncoder()
    }

    scaler = StandardScaler()

    def preprocess(df):
        df_copy = df.copy()

        # Drop unwanted columns if they exist
        drop_cols_iter = [c for c in cols_to_drop if c in df_copy.columns]
        df_copy = df_copy.drop(drop_cols_iter, axis=1)

        # Label encode categorical columns
        for col, le in col_encoders.items():
            if col in df_copy.columns:
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))

        # Standard scale Lesion_Volume if present
        if 'Lesion_Volume' in df_copy.columns:
            df_copy['Lesion_Volume'] = scaler.fit_transform(
                df_copy[['Lesion_Volume']]
            )

        print(f"{df_copy.shape}, missing values: {df_copy.isna().sum().sum()}")
        return df_copy

    processed = {key: preprocess(df) for key, df in datasets.items()}
    return processed

def generate_meta(datasets):
    """
    Generate a row-aligned meta dictionary containing identifiers
    for each dataset, so you can safely do subject-level analyses.
    """
    meta_dict = {}
    for key, df in datasets.items():
        # Keep only identifiers
        cols = [c for c in ["SubjID", "Agent_Noun"] if c in df.columns]
        meta_dict[key] = df[cols].reset_index(drop=True)
    return meta_dict
