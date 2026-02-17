"""
Iris Dataset Exploration Script
Phase 2: Data Exploration

This script performs basic data exploration on the Iris dataset:
- Load the dataset
- Understand structure
- Check for missing values
- Generate statistics
- Create summary report
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

def load_and_explore_iris():
    """
    Main function to load and explore the Iris dataset.
    This function performs all the exploration steps and saves results.
    """
    
    print("=" * 60)
    print("PHASE 2: IRIS DATASET EXPLORATION")
    print("=" * 60)
    
    # ----------------------------------------------------------------------
    # STEP 1: Load the dataset
    # ----------------------------------------------------------------------
    print("\n STEP 1: Loading the Iris dataset")
    print("-" * 40)
    
    # Load data from sklearn
    iris = load_iris()
    
    # Extract components
    data = iris.data
    target = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Features: {feature_names}")
    print(f"  - Target classes: {target_names}")
    
    # ----------------------------------------------------------------------
    # STEP 2: Create DataFrame
    # ----------------------------------------------------------------------
    print("\n STEP 2: Creating DataFrame")
    print("-" * 40)
    
    # Create DataFrame with feature columns
    df = pd.DataFrame(data, columns=feature_names)
    
    # Add target columns
    df['species_code'] = target
    df['species_name'] = df['species_code'].map({
        0: target_names[0], 
        1: target_names[1], 
        2: target_names[2]
    })
    
    print(f"✓ DataFrame created with shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # ----------------------------------------------------------------------
    # STEP 3: Check data types and structure
    # ----------------------------------------------------------------------
    print("\n STEP 3: Data Types and Structure")
    print("-" * 40)
    
    print("Data types:")
    print(df.dtypes)
    
    print("\nDataset info:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Total features: {len(feature_names)}")
    print(f"  - Target classes: {len(target_names)}")
    
    # ----------------------------------------------------------------------
    # STEP 4: Check for missing values
    # ----------------------------------------------------------------------
    print("\n STEP 4: Missing Values Check")
    print("-" * 40)
    
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("\n✓ GOOD NEWS: No missing values found!")
    else:
        print(f"\n⚠ WARNING: Found {missing_values.sum()} missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # ----------------------------------------------------------------------
    # STEP 5: Generate basic statistics
    # ----------------------------------------------------------------------
    print("\ STEP 5: Basic Statistics")
    print("-" * 40)
    
    # Statistics for numerical features
    print("Statistics for numerical features:")
    stats = df[feature_names].describe()
    print(stats)
    
    # Statistics by species
    print("\nStatistics grouped by species:")
    grouped_stats = df.groupby('species_name')[feature_names].describe()
    print(grouped_stats)
    
    # Class distribution
    print("\nClass distribution:")
    class_dist = df['species_name'].value_counts()
    print(class_dist)
    
    # ----------------------------------------------------------------------
    # STEP 6: Create summary report
    # ----------------------------------------------------------------------
    print("\n STEP 6: Creating Summary Report")
    print("-" * 40)
    
    # Create a comprehensive summary report
    report = []
    report.append("=" * 60)
    report.append("IRIS DATASET - COMPLETE SUMMARY REPORT")
    report.append("=" * 60)
    
    # Dataset overview
    report.append("\n1. DATASET OVERVIEW")
    report.append(f"   - Total samples: {len(df)}")
    report.append(f"   - Number of features: {len(feature_names)}")
    report.append(f"   - Feature names: {', '.join(feature_names)}")
    report.append(f"   - Number of classes: {len(target_names)}")
    report.append(f"   - Class names: {', '.join(target_names)}")
    
    # Class distribution
    report.append("\n2. CLASS DISTRIBUTION")
    for species, count in class_dist.items():
        percentage = (count / len(df)) * 100
        report.append(f"   - {species}: {count} samples ({percentage:.1f}%)")
    
    # Data quality
    report.append("\n3. DATA QUALITY")
    report.append(f"   - Missing values: {missing_values.sum()}")
    report.append(f"   - Duplicate rows: {duplicates}")
    report.append(f"   - Data types: All numerical features are float64")
    
    # Feature statistics
    report.append("\n4. FEATURE STATISTICS (min - max)")
    for feature in feature_names:
        report.append(f"   - {feature}: {df[feature].min():.2f} - {df[feature].max():.2f} cm")
    
    # Key insights
    report.append("\n5. KEY INSIGHTS")
    report.append("   ✓ Dataset is perfectly balanced (50 samples per class)")
    report.append("   ✓ No missing values - data is clean and ready for ML")
    report.append("   ✓ Features are on different scales:")
    report.append("      - Sepal measurements: ~4-8 cm")
    report.append("      - Petal measurements: ~1-7 cm")
    report.append("   ✓ Petal measurements show more variation than sepal measurements")
    report.append("   ✓ This suggests petal features might be better for classification")
    
    report.append("\n" + "=" * 60)
    
    # Print the report
    report_text = "\n".join(report)
    print(report_text)
    
    # ----------------------------------------------------------------------
    # STEP 7: Save results to files
    # ----------------------------------------------------------------------
    print("\n STEP 7: Saving Results")
    print("-" * 40)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save DataFrame to CSV
    csv_path = 'data/iris_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Dataset saved to: {csv_path}")
    
    # Save summary report to text file
    report_path = 'data/dataset_summary.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"✓ Summary report saved to: {report_path}")
    
    # Save basic statistics to CSV
    stats_path = 'data/basic_statistics.csv'
    stats.to_csv(stats_path)
    print(f"✓ Statistics saved to: {stats_path}")
    
    print("\n" + "=" * 60)
    print(" PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return df, feature_names, target_names

def quick_summary():
    """
    Quick function to get a summary of the dataset
    Useful when you need to quickly check something
    """
    iris = load_iris()
    print(f"Iris Dataset Summary:")
    print(f"- Samples: {iris.data.shape[0]}")
    print(f"- Features: {iris.data.shape[1]}")
    print(f"- Classes: {iris.target_names}")
    print(f"- Feature names: {iris.feature_names}")

# This code runs only when you execute this script directly
if __name__ == "__main__":
    # Run the full exploration
    df, features, targets = load_and_explore_iris()
    
    # You can add any additional analysis here
    print("\n" + "=" * 60)
    print("Additional Analysis:")
    print("=" * 60)
    
    # Example: Find flowers with extreme measurements
    print("\nFlowers with largest sepal length:")
    largest_sepal = df.nlargest(3, 'sepal length (cm)')[['sepal length (cm)', 'species_name']]
    print(largest_sepal)
    
    print("\nFlowers with smallest petal width:")
    smallest_petal = df.nsmallest(3, 'petal width (cm)')[['petal width (cm)', 'species_name']]
    print(smallest_petal)