"""Data loading module for Iris dataset."""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

def load_iris_data():
    """
    Load the Iris dataset and return as DataFrame.
    
    Returns:
        tuple: (X, y, feature_names, target_names, df)
        - X: Feature matrix (numpy array)
        - y: Target vector (numpy array)
        - feature_names: List of feature names
        - target_names: List of target class names
        - df: Combined DataFrame for easy exploration
    """
    # Load the dataset from sklearn
    iris = load_iris()
    
    # Extract components
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create a pandas DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({i: name for i, name in enumerate(target_names)})
    
    print("✓ Iris dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Target classes: {len(target_names)}")
    print(f"  - Total samples: {len(df)}")
    
    return X, y, feature_names, target_names, df

def get_dataset_info(df):
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Iris dataset DataFrame
    
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
    }
    return info

if __name__ == "__main__":
    # Test the loader
    X, y, feature_names, target_names, df = load_iris_data()
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    info = get_dataset_info(df)
    print(f"Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing values: {info['missing_values']}")