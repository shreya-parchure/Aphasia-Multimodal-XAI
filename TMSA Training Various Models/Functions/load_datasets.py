import pandas as pd
import numpy as np

def load(**dataset_paths):
    
    def read(file_name):
        dataset = pd.read_excel(file_name)
        print(f"Loaded {file_name}, shape: {dataset.shape}")
        return dataset
    
    datasets = {}
    for name, path in dataset_paths.items():
        datasets[name] = read(path)
    
    return datasets
