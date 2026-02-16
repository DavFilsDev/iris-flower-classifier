"""
Data exploration script for Iris dataset.
This script performs all the exploration steps programmatically.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

def load_and_explore_iris():
    """Main function to load and explore the Iris dataset."""
    
    print("=" * 60)
    print("IRIS DATASET EXPLORATION")
    print("=" * 60)
    
    # 1. LOAD THE DATASET
    print("\n📌 1. LOADING THE DATASET")
    print("-" * 40)
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({i: name for i, name in enumerate(target_names)})
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {feature_names}")
    print(f"  - Target classes: {target_names}")
    print(f"  - Total samples: {len(df)}")
    
    # 2. UNDERSTAND DATA STRUCTURE
    print("\n📌 2. DATA STRUCTURE")
    print("-" * 40)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    # 3. CHECK FOR MISSING VALUES
    print("\n📌 3. MISSING VALUES CHECK")
    print("-" * 40)
    
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found! The dataset is clean.")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # 4. GENERATE BASIC STATISTICS
    print("\n📌 4. BASIC STATISTICS")
    print("-" * 40)
    
    print("\nOverall statistics:")
    print(df[feature_names].describe())
    
    print("\nStatistics by species:")
    print(df.groupby('species_name')[feature_names].describe())
    
    # 5. CREATE SUMMARY REPORT
    print("\n📌 5. SUMMARY REPORT")
    print("-" * 40)
    
    report = []
    report.append("=" * 60)
    report.append("IRIS DATASET SUMMARY REPORT")
    report.append("=" * 60)
    
    # Class distribution
    report.append("\n📊 CLASS DISTRIBUTION:")
    class_counts = df['species_name'].value_counts()
    for species, count in class_counts.items():
        percentage = (count/len(df))*100
        report.append(f"  - {species}: {count} samples ({percentage:.1f}%)")
    
    # Feature ranges
    report.append("\n📏 FEATURE RANGES (min - max):")
    for feature in feature_names:
        report.append(f"  - {feature}: {df[feature].min():.2f} - {df[feature].max():.2f} cm")
    
    # Key insights
    report.append("\n💡 KEY INSIGHTS:")
    report.append("  ✓ Dataset is perfectly balanced (50 samples per class)")
    report.append("  ✓ No missing values or data quality issues")
    report.append("  ✓ Features are on different scales (need normalization for some ML algorithms)")
    report.append("  ✓ Petal measurements show more variation than sepal measurements")
    
    report.append("\n" + "=" * 60)
    
    # Print the report
    print("\n".join(report))
    
    # Save the report
    os.makedirs('../data', exist_ok=True)
    with open('../data/dataset_summary.txt', 'w') as f:
        f.write("\n".join(report))
    print("\n✓ Report saved to 'data/dataset_summary.txt'")
    
    return df, feature_names, target_names

if __name__ == "__main__":
    # Run the exploration
    df, features, targets = load_and_explore_iris()